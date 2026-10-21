"""Job task: silver_jobs → matching text → GTE embedding → silver_job_embeddings."""

import argparse
from datetime import UTC, datetime

from job_hunter import llm, matching, storage
from job_hunter.records import JobEmbedding, Seniority, WorkMode
from job_hunter.storage import Table


def main():
    parser = argparse.ArgumentParser(description="Embed enriched jobs for matching")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--embedding-model", default=llm.DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--max-jobs", type=int, default=500)
    args = parser.parse_args()

    from databricks.sdk.runtime import spark

    storage.ensure_tables(spark, args.catalog, args.schema)

    def name(table: Table) -> str:
        return storage.full_name(args.catalog, args.schema, table)

    target = name(Table.SILVER_JOB_EMBEDDINGS)

    # Jobs never embedded, or re-enriched since they were embedded.
    pending = spark.sql(
        f"""
        SELECT s.job_id, s.title_en, s.seniority, s.work_mode, s.country, s.country_code, s.skills, s.summary,
               p.company, p.url, f.first_seen_at
        FROM {name(Table.SILVER_JOBS)} s
        JOIN {name(Table.POSTINGS)} p ON p.job_id = s.job_id
        LEFT JOIN (
            SELECT job_id, min(received_at) AS first_seen_at
            FROM (SELECT inline(job_cards), received_at FROM {name(Table.EMAILS)})
            GROUP BY job_id
        ) f ON f.job_id = s.job_id
        LEFT JOIN {target} e ON e.job_id = s.job_id
        WHERE e.job_id IS NULL OR e.embedded_at < s.enriched_at
        LIMIT {int(args.max_jobs)}
        """
    ).collect()
    print(f"{len(pending)} job(s) to embed with {args.embedding_model}")
    if not pending:
        return

    texts = [
        matching.match_text(r.title_en, r.seniority, r.work_mode, r.country, r.skills or [], r.summary)
        for r in pending
    ]
    vectors = llm.embed(texts, model=args.embedding_model)

    now = datetime.now(UTC)
    rows = [
        JobEmbedding(
            job_id=r.job_id,
            match_text=text,
            embedding=vector,
            embedding_model=args.embedding_model,
            title_en=r.title_en,
            company=r.company,
            url=r.url,
            country_code=r.country_code,
            work_mode=WorkMode(r.work_mode),
            seniority=Seniority(r.seniority),
            first_seen_at=r.first_seen_at,
            embedded_at=now,
        )
        for r, text, vector in zip(pending, texts, vectors, strict=True)
    ]
    storage.merge(spark, storage.to_dataframe(spark, Table.SILVER_JOB_EMBEDDINGS, rows), target, ["job_id"],
                  update_when="true")
    print(f"Embedded {len(rows)} job(s) into {target}")


if __name__ == "__main__":
    main()
