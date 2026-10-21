"""Job task: CV PDF in a UC volume → structured JSON + embedding → silver_cv."""

import argparse
from datetime import UTC, datetime

from job_hunter import cv, llm, storage


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
    table = storage.full_name(args.catalog, args.schema, storage.SILVER_CV_TABLE)

    with open(args.cv_path, "rb") as f:
        pdf_bytes = f.read()
    version = cv.file_version(pdf_bytes)

    if not args.force and spark.sql(f"SELECT 1 FROM {table} WHERE cv_version = '{version}'").count():
        print(f"CV version {version} already processed; nothing to do")
        return

    text = cv.extract_text(pdf_bytes)
    print(f"Extracted {len(text)} characters from {args.cv_path} (version {version})")

    structured = cv.normalize(llm.chat_json(cv.SYSTEM_PROMPT, text, cv.SCHEMA, model=args.model, max_tokens=16000))
    print(f"Structured CV: {len(structured['experience'])} roles, {len(structured['certifications'])} certifications")
    if missing := cv.roles_without_bullets(structured):
        print(f"WARNING: roles without bullet points (check the PDF layout): {missing}")

    [embedding] = llm.embed([cv.match_text(structured)], model=args.embedding_model)
    row = cv.cv_row(version, args.cv_path, text, structured, embedding, args.model, args.embedding_model,
                    datetime.now(UTC))

    storage.merge(spark, spark.createDataFrame([row], storage.SILVER_CV_SCHEMA), table, ["cv_version"],
                  update_when="true")
    # Only one CV version is current; matching always uses it.
    spark.sql(f"UPDATE {table} SET is_current = false WHERE is_current AND cv_version <> '{version}'")
    print(f"Stored CV version {version} as current in {table}")


if __name__ == "__main__":
    main()
