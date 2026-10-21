"""Job task: job ids from bronze emails → LinkedIn job descriptions → bronze_job_postings."""

import argparse
import time
from http import HTTPStatus

import requests

from job_hunter import application_status, linkedin_jobs, storage
from job_hunter.records import FINAL_HTTP_STATUSES, ApplyType, StatusValue
from job_hunter.storage import Table


def main():
    parser = argparse.ArgumentParser(description="Fetch LinkedIn job descriptions into bronze")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--max-postings", type=int, default=100)
    parser.add_argument("--delay-seconds", type=float, default=2.0)
    parser.add_argument("--max-recheck", type=int, default=40,
                        help="Postings per run re-read to detect closed jobs and their apply type")
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

    rate_limited = any(p.http_status == HTTPStatus.TOO_MANY_REQUESTS for p in postings)
    if not rate_limited and args.max_recheck > 0:
        recheck_open_jobs(spark, args, postings_table, http)


def recheck_open_jobs(spark, args, postings_table: str, http: requests.Session) -> None:
    """Re-read the pages of jobs that can still reach the ranking, to catch postings that closed.

    Candidates, most important first: jobs with a generated application, judged jobs that are not excluded,
    then postings never checked for their apply type. A job is re-checked at most once every 20 hours and
    only when it has no status yet. Only apply_type / apply_checked_at change (fetched_at stays), so nothing
    is re-enriched. Closed jobs get the status 'closed', which removes them from future rankings.
    """
    def name(table: Table) -> str:
        return storage.full_name(args.catalog, args.schema, table)

    ids = [
        r.job_id
        for r in spark.sql(f"""
            SELECT p.job_id
            FROM {postings_table} p
            LEFT JOIN (SELECT DISTINCT job_id FROM {name(Table.GOLD_APPLICATIONS)}) a ON a.job_id = p.job_id
            LEFT JOIN (SELECT job_id, max(fit_score) AS fit FROM {name(Table.GOLD_JOB_MATCHES)}
                       WHERE priority <> 'excluded' GROUP BY job_id) m ON m.job_id = p.job_id
            LEFT ANTI JOIN {name(Table.GOLD_APPLICATION_STATUS)} st ON st.job_id = p.job_id
            WHERE p.http_status = 200
              AND (p.apply_checked_at IS NULL OR p.apply_checked_at < current_timestamp() - INTERVAL 20 HOURS)
              AND (a.job_id IS NOT NULL OR m.job_id IS NOT NULL OR p.apply_type IS NULL)
            ORDER BY a.job_id IS NULL, m.fit DESC NULLS LAST, p.fetched_at DESC
            LIMIT {int(args.max_recheck)}
        """).collect()
    ]
    if not ids:
        return
    updates = []
    for job_id in ids:
        posting = linkedin_jobs.fetch_job_posting(job_id, http)
        if posting.http_status == HTTPStatus.TOO_MANY_REQUESTS:
            print("Rate limited by LinkedIn during the re-check; continuing next run")
            break
        if posting.ok:
            updates.append(posting)
        time.sleep(args.delay_seconds)
    if not updates:
        return
    source = spark.createDataFrame([(p.job_id, p.apply_type.value, p.apply_checked_at) for p in updates],
                                   "job_id STRING, apply_type STRING, apply_checked_at TIMESTAMP")
    storage.update_columns(spark, source, postings_table, keys=["job_id"], columns=["apply_type", "apply_checked_at"])
    closed = [p.job_id for p in updates if p.apply_type is ApplyType.CLOSED]
    application_status.save(spark, args.catalog, args.schema, application_status.make_statuses(
        closed, StatusValue.CLOSED, "pipeline", "LinkedIn no longer accepts applications"))
    print(f"Re-checked {len(updates)} of {len(ids)} posting(s); {len(closed)} closed")


if __name__ == "__main__":
    main()
