"""Job task: CV PDF in a UC volume → ai_parse_document layout elements → validated CvDocument + embedding → silver_cv."""

import argparse
from datetime import UTC, datetime

from job_hunter import cv, llm, storage
from job_hunter.storage import Table


def main():
    parser = argparse.ArgumentParser(description="Parse the CV PDF into silver_cv")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--cv-path", required=True, help="e.g. /Volumes/<catalog>/<schema>/cv/input/cv.pdf")
    parser.add_argument("--model", default=llm.DEFAULT_CHAT_MODEL)
    parser.add_argument("--embedding-model", default=llm.DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--force", action="store_true", help="Reprocess even if this file version exists")
    args = parser.parse_args()

    from databricks.sdk.runtime import spark

    storage.ensure_tables(spark, args.catalog, args.schema)
    table = storage.full_name(args.catalog, args.schema, Table.SILVER_CV)

    with open(args.cv_path, "rb") as f:
        pdf_bytes = f.read()
    version = cv.file_version(pdf_bytes)

    if not args.force and spark.sql(f"SELECT 1 FROM {table} WHERE cv_version = '{version}'").count():
        print(f"CV version {version} already processed; nothing to do")
        return

    rows = [r.asDict() for r in spark.sql(cv.parse_document_sql(), args={"path": args.cv_path}).collect()]
    if errors := {r["error"] for r in rows if r["error"]}:
        raise RuntimeError(f"ai_parse_document failed for {args.cv_path}: {errors}")
    elements = [r for r in rows if r["pos"] is not None]
    text = cv.elements_text(elements)
    if not text:
        raise RuntimeError(f"ai_parse_document found no text in {args.cv_path}")
    print(f"Parsed {len(elements)} layout elements on {max(e['page'] or 1 for e in elements)} page(s) into "
          f"{len(text)} characters from {args.cv_path} (version {version})")
    if unsure := cv.low_confidence(elements):
        print(f"WARNING: {len(unsure)} element(s) parsed with low confidence: "
              f"{[(e['page'], (e['content'] or '')[:60]) for e in unsure]}")

    document = llm.chat_json(cv.SYSTEM_PROMPT, text, cv.CvDocument, model=args.model, max_tokens=16000)
    print(f"Structured CV: {len(document.experience)} roles, {len(document.certifications)} certifications")
    if missing := document.roles_without_bullets():
        print(f"WARNING: roles without bullet points (check the PDF layout): {missing}")

    [embedding] = llm.embed([document.match_text()], model=args.embedding_model)
    record = cv.to_cv_version(version, args.cv_path, text, document, embedding, args.model, args.embedding_model,
                              datetime.now(UTC))

    storage.merge(spark, storage.to_dataframe(spark, Table.SILVER_CV, [record]), table, ["cv_version"],
                  update_when="true")
    # Only one CV version is current; matching always uses it.
    spark.sql(f"UPDATE {table} SET is_current = false WHERE is_current AND cv_version <> '{version}'")
    print(f"Stored CV version {version} as current in {table}")


if __name__ == "__main__":
    main()
