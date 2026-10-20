"""Job task: job ids from bronze emails → LinkedIn job descriptions → bronze_job_postings."""

import argparse
import time

import requests

from job_hunter import linkedin_jobs, storage


def posting_row(posting: dict) -> tuple:
    """Flatten a `linkedin_jobs.fetch_job_posting` result into a row in POSTINGS_SCHEMA column order."""
    return (
        posting["job_id"],
        posting["url"],
        posting.get("title"),
        posting.get("company"),
        posting.get("location"),
        posting.get("description"),
        posting.get("criteria"),
        posting["http_status"],
        posting["fetched_at"],
    )


def main():
    parser = argparse.ArgumentParser(description="Fetch LinkedIn job descriptions into bronze")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--max-postings", type=int, default=100)
    parser.add_argument("--delay-seconds", type=float, default=2.0)
    args = parser.parse_args()

    from databricks.sdk.runtime import spark

    storage.ensure_tables(spark, args.catalog, args.schema)
    emails = storage.full_name(args.catalog, args.schema, storage.EMAILS_TABLE)
    postings = storage.full_name(args.catalog, args.schema, storage.POSTINGS_TABLE)

    # Newest job ids without a final fetch outcome yet (read OK, or removed by LinkedIn).
    final = ", ".join(str(s) for s in storage.FINAL_HTTP_STATUSES)
    pending = [
        r.job_id
        for r in spark.sql(f"""
            SELECT c.job_id, max(c.received_at) AS last_seen
            FROM (SELECT inline(job_cards), received_at FROM {emails}) c
            LEFT ANTI JOIN (SELECT job_id FROM {postings} WHERE http_status IN ({final})) p
              ON p.job_id = c.job_id
            GROUP BY c.job_id
            ORDER BY last_seen DESC
            LIMIT {args.max_postings}
        """).collect()
    ]
    print(f"{len(pending)} job posting(s) to fetch")

    http = requests.Session()
    rows = []
    for job_id in pending:
        posting = linkedin_jobs.fetch_job_posting(job_id, http)
        rows.append(posting_row(posting))
        if posting["http_status"] == 429:
            print("Rate limited by LinkedIn; stopping early, the rest is retried next run")
            break
        time.sleep(args.delay_seconds)

    if rows:
        storage.merge(
            spark,
            spark.createDataFrame(rows, storage.POSTINGS_SCHEMA),
            postings,
            ["job_id"],
            update_when=f"t.http_status NOT IN ({final})",
        )

    ok = sum(1 for r in rows if r[7] == 200)
    removed = sum(1 for r in rows if r[7] == 404)
    print(f"Fetched {len(rows)} posting(s): {ok} OK, {removed} removed (404), {len(rows) - ok - removed} failed")
    if rows and ok + removed == 0:
        raise RuntimeError("No job posting could be fetched; LinkedIn may be blocking requests")


if __name__ == "__main__":
    main()
