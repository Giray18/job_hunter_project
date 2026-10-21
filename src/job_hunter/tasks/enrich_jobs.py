"""Job task: bronze_job_postings → LLM translation + field extraction → silver_jobs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from job_hunter import countries, enrichment, llm, storage
from job_hunter.records import JobPosting, SilverJob
from job_hunter.storage import Table


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
    postings_table = storage.full_name(args.catalog, args.schema, Table.POSTINGS)
    silver = storage.full_name(args.catalog, args.schema, Table.SILVER_JOBS)

    backfill_country_codes(spark, silver)

    # Postings never enriched, or re-fetched since they were enriched.
    pending = [
        JobPosting(**r.asDict())
        for r in spark.sql(f"""
            SELECT p.*
            FROM {postings_table} p
            LEFT JOIN {silver} s ON s.job_id = p.job_id
            WHERE p.http_status = 200 AND p.description IS NOT NULL
              AND (s.job_id IS NULL OR s.source_fetched_at < p.fetched_at)
            ORDER BY p.fetched_at DESC
            LIMIT {args.max_jobs}
        """).collect()
    ]
    print(f"{len(pending)} posting(s) to enrich with {args.model}")

    def enrich(posting: JobPosting) -> SilverJob | None:
        try:
            result = llm.chat_json(enrichment.SYSTEM_PROMPT, enrichment.user_prompt(posting),
                                   enrichment.EnrichedJob, model=args.model)
        except RuntimeError as e:
            print(f"job {posting.job_id}: {e}")
            return None
        return enrichment.to_silver_job(posting, result, args.model, datetime.now(UTC))

    with ThreadPoolExecutor(max_workers=args.parallelism) as pool:
        jobs = [job for job in pool.map(enrich, pending) if job]

    if jobs:
        storage.merge(spark, storage.to_dataframe(spark, Table.SILVER_JOBS, jobs), silver, ["job_id"],
                      update_when="s.enriched_at > t.enriched_at")
    print(f"Enriched {len(jobs)} of {len(pending)} posting(s) into {silver}")
    if pending and not jobs:
        raise RuntimeError("No posting could be enriched; check the model endpoint")


def backfill_country_codes(spark, silver: str) -> None:
    """Rows enriched before country codes existed: derive the ISO code from the stored country name."""
    rows = spark.sql(f"SELECT job_id, country FROM {silver} WHERE country_code IS NULL AND country IS NOT NULL")
    updates = [
        (r.job_id, countries.name(code), code)
        for r in rows.collect()
        if (code := countries.code_from_name(r.country))
    ]
    if not updates:
        return
    source = spark.createDataFrame(updates, "job_id STRING, country STRING, country_code STRING")
    storage.update_columns(spark, source, silver, keys=["job_id"], columns=["country", "country_code"])
    print(f"Backfilled ISO country codes for {len(updates)} row(s)")


if __name__ == "__main__":
    main()
