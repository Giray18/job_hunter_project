"""Job task: bronze_job_postings → LLM translation + field extraction → silver_jobs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from job_hunter import enrichment, llm, storage


def main():
    parser = argparse.ArgumentParser(description="Translate and enrich job postings with an LLM")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--model", default=llm.DEFAULT_CHAT_MODEL)
    parser.add_argument("--max-jobs", type=int, default=200)
    # Pay-per-token endpoints have a per-workspace tokens-per-minute limit; keep this low.
    parser.add_argument("--parallelism", type=int, default=2)
    args = parser.parse_args()

    from databricks.sdk.runtime import spark

    storage.ensure_tables(spark, args.catalog, args.schema)
    postings = storage.full_name(args.catalog, args.schema, storage.POSTINGS_TABLE)
    silver = storage.full_name(args.catalog, args.schema, storage.SILVER_JOBS_TABLE)

    # Postings never enriched, or re-fetched since they were enriched.
    pending = [
        r.asDict()
        for r in spark.sql(f"""
            SELECT p.job_id, p.title, p.company, p.location, p.criteria, p.description, p.fetched_at
            FROM {postings} p
            LEFT JOIN {silver} s ON s.job_id = p.job_id
            WHERE p.http_status = 200 AND p.description IS NOT NULL
              AND (s.job_id IS NULL OR s.source_fetched_at < p.fetched_at)
            ORDER BY p.fetched_at DESC
            LIMIT {args.max_jobs}
        """).collect()
    ]
    print(f"{len(pending)} posting(s) to enrich with {args.model}")

    def enrich(posting: dict):
        try:
            result = llm.chat_json(enrichment.SYSTEM_PROMPT, enrichment.user_prompt(posting), enrichment.SCHEMA,
                                   model=args.model)
        except RuntimeError as e:
            print(f"job {posting['job_id']}: {e}")
            return None
        return enrichment.enriched_row(posting["job_id"], result, posting["description"], args.model,
                                       posting["fetched_at"], datetime.now(UTC))

    with ThreadPoolExecutor(max_workers=args.parallelism) as pool:
        rows = [row for row in pool.map(enrich, pending) if row]

    if rows:
        storage.merge(spark, spark.createDataFrame(rows, storage.SILVER_JOBS_SCHEMA), silver, ["job_id"],
                      update_when="s.enriched_at > t.enriched_at")
    print(f"Enriched {len(rows)} of {len(pending)} posting(s) into {silver}")
    if pending and not rows:
        raise RuntimeError("No posting could be enriched; check the model endpoint")


if __name__ == "__main__":
    main()
