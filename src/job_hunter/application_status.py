"""Application status per job (applied / skipped / closed / not_eligible).

Written by the pipeline (closed postings), by the browser application run and manually. Any status removes the
job from rankings, so it never comes back in the daily top N.

Local usage (writes to Databricks through Databricks Connect):
    uv run python -m job_hunter.application_status --status applied --job-ids 4479821741,4478252899
    uv run python -m job_hunter.application_status --status closed --job-ids 4476677082 --notes "closed on LinkedIn"
    uv run python -m job_hunter.application_status --schema dev_<you>_medallion --profile personal-projects ...
"""

import argparse
import os
from datetime import UTC, datetime

from pyspark.sql import SparkSession

from job_hunter import storage
from job_hunter.records import ApplicationStatus, StatusValue
from job_hunter.storage import Table

# "applied" is final: a later "closed" (the posting closed after the user applied) must not overwrite it.
UPDATE_WHEN = f"t.status <> '{StatusValue.APPLIED.value}' OR s.status = '{StatusValue.APPLIED.value}'"


def make_statuses(job_ids: list[str], status: StatusValue, source: str, notes: str | None = None,
                  now: datetime | None = None) -> list[ApplicationStatus]:
    now = now or datetime.now(UTC)
    unique = list(dict.fromkeys(j.strip() for j in job_ids if j and j.strip()))
    return [ApplicationStatus(job_id=j, status=status, source=source, notes=notes, updated_at=now) for j in unique]


def save(spark: SparkSession, catalog: str, schema: str, statuses: list[ApplicationStatus]) -> None:
    if not statuses:
        return
    target = storage.full_name(catalog, schema, Table.GOLD_APPLICATION_STATUS)
    storage.merge(spark, storage.to_dataframe(spark, Table.GOLD_APPLICATION_STATUS, statuses), target, ["job_id"],
                  update_when=UPDATE_WHEN)


def main():
    parser = argparse.ArgumentParser(description="Record application status for jobs")
    parser.add_argument("--job-ids", required=True, help="Comma-separated LinkedIn job ids")
    parser.add_argument("--status", required=True, type=StatusValue, choices=list(StatusValue))
    parser.add_argument("--notes")
    parser.add_argument("--source", default="manual", choices=["manual", "browser", "pipeline"])
    parser.add_argument("--catalog", default="job_hunter_project")
    parser.add_argument("--schema", default="medallion")
    parser.add_argument("--profile", default=os.environ.get("DATABRICKS_CONFIG_PROFILE"))
    args = parser.parse_args()

    if args.profile:
        os.environ["DATABRICKS_CONFIG_PROFILE"] = args.profile
    from databricks.connect import DatabricksSession

    spark = DatabricksSession.builder.serverless().getOrCreate()
    # Only the status table: other prod tables belong to the pipeline's service principal.
    storage.ensure_tables(spark, args.catalog, args.schema, only=[Table.GOLD_APPLICATION_STATUS])
    statuses = make_statuses(args.job_ids.split(","), args.status, args.source, args.notes)
    save(spark, args.catalog, args.schema, statuses)
    print(f"Recorded '{args.status}' for {len(statuses)} job(s) in {args.catalog}.{args.schema}")


if __name__ == "__main__":
    main()
