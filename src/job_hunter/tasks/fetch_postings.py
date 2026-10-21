"""Job task: job ids from bronze emails → LinkedIn job descriptions → bronze_job_postings."""

import argparse
import time
from http import HTTPStatus

import requests

from job_hunter import linkedin_jobs, storage
from job_hunter.records import FINAL_HTTP_STATUSES
from job_hunter.storage import Table


def main():
    parser = argparse.ArgumentParser(description="Fetch LinkedIn job descriptions into bronze")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--max-postings", type=int, default=100)
    parser.add_argument("--delay-seconds", type=float, default=2.0)
    args = parser.parse_args()

    from databricks.sdk.runtime import spark

    storage.ensure_tables(spark, args.catalog, args.schema)
    emails = storage.full_name(args.catalog, args.schema, Table.EMAILS)
    postings_table = storage.full_name(args.catalog, args.schema, Table.POSTINGS)

    # Newest job ids without a final fetch outcome yet (read OK, or removed by LinkedIn).
    final = ", ".join(str(int(s)) for s in FINAL_HTTP_STATUSES)
    pending = [
        r.job_id
        for r in spark.sql(f"""
            SELECT c.job_id, max(c.received_at) AS last_seen
            FROM (SELECT inline(job_cards), received_at FROM {emails}) c
            LEFT ANTI JOIN (SELECT job_id FROM {postings_table} WHERE http_status IN ({final})) p
              ON p.job_id = c.job_id
            GROUP BY c.job_id
            ORDER BY last_seen DESC
            LIMIT {args.max_postings}
        """).collect()
    ]
    print(f"{len(pending)} job posting(s) to fetch")

    http = requests.Session()
    postings = []
    for job_id in pending:
        posting = linkedin_jobs.fetch_job_posting(job_id, http)
        postings.append(posting)
        if posting.http_status == HTTPStatus.TOO_MANY_REQUESTS:
            print("Rate limited by LinkedIn; stopping early, the rest is retried next run")
            break
        time.sleep(args.delay_seconds)

    if postings:
        storage.merge(
            spark,
            storage.to_dataframe(spark, Table.POSTINGS, postings),
            postings_table,
            ["job_id"],
            update_when=f"t.http_status NOT IN ({final})",
        )

    ok = sum(1 for p in postings if p.ok)
    removed = sum(1 for p in postings if p.http_status == HTTPStatus.NOT_FOUND)
    print(f"Fetched {len(postings)} posting(s): {ok} OK, {removed} removed (404), {len(postings) - ok - removed} failed")
    if postings and ok + removed == 0:
        raise RuntimeError("No job posting could be fetched; LinkedIn may be blocking requests")


if __name__ == "__main__":
    main()
