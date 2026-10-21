"""Job task: LinkedIn job-alert emails from Gmail → bronze_job_alert_emails."""

import argparse
import dataclasses
from datetime import UTC, datetime

from pyspark.sql.functions import col

from job_hunter import gmail_client, storage
from job_hunter.databricks_secrets import get_secret
from job_hunter.storage import Table


def main():
    parser = argparse.ArgumentParser(description="Ingest LinkedIn job-alert emails into bronze")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--secret-scope", required=True)
    parser.add_argument("--query", default=gmail_client.LINKEDIN_JOB_ALERTS_QUERY)
    parser.add_argument("--max-messages", type=int, default=200)
    args = parser.parse_args()

    from databricks.sdk.runtime import spark

    storage.ensure_tables(spark, args.catalog, args.schema)
    table = storage.full_name(args.catalog, args.schema, Table.EMAILS)

    creds = gmail_client.credentials_from_refresh_token(
        get_secret(args.secret_scope, "gmail_client_id"),
        get_secret(args.secret_scope, "gmail_client_secret"),
        get_secret(args.secret_scope, "gmail_refresh_token"),
    )
    service = gmail_client.build_service(creds)

    ids = gmail_client.list_message_ids(service, args.query, args.max_messages)
    known = set()
    if ids:
        known = {r.message_id for r in spark.table(table).where(col("message_id").isin(ids)).collect()}
    new_ids = [i for i in ids if i not in known]
    print(f"Query '{args.query}': {len(ids)} message(s), {len(new_ids)} new")

    now = datetime.now(UTC)
    emails = [dataclasses.replace(gmail_client.get_message(service, i), ingested_at=now) for i in new_ids]
    if emails:
        storage.merge(spark, storage.to_dataframe(spark, Table.EMAILS, emails), table, ["message_id"])
    print(f"Merged {len(emails)} email(s) with {sum(len(e.job_cards) for e in emails)} job card(s) into {table}")


if __name__ == "__main__":
    main()
