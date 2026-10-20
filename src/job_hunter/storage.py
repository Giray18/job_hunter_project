"""Bronze table definitions and an idempotent MERGE helper."""

import uuid

from pyspark.sql import DataFrame, SparkSession

EMAILS_TABLE = "bronze_job_alert_emails"
POSTINGS_TABLE = "bronze_job_postings"

JOB_CARD_TYPE = "STRUCT<job_id: STRING, url: STRING, title: STRING, company: STRING, location: STRING>"

EMAILS_SCHEMA = f"""
    message_id STRING NOT NULL,
    thread_id STRING,
    received_at TIMESTAMP,
    sender STRING,
    subject STRING,
    snippet STRING,
    text STRING,
    job_cards ARRAY<{JOB_CARD_TYPE}>,
    ingested_at TIMESTAMP
"""

POSTINGS_SCHEMA = """
    job_id STRING NOT NULL,
    url STRING,
    title STRING,
    company STRING,
    location STRING,
    description STRING,
    criteria MAP<STRING, STRING>,
    http_status INT,
    fetched_at TIMESTAMP
"""

_TABLES = {
    EMAILS_TABLE: (EMAILS_SCHEMA, "LinkedIn job-alert emails from Gmail (URL query strings stripped)."),
    POSTINGS_TABLE: (POSTINGS_SCHEMA, "LinkedIn job postings ('About the job' text and criteria) by job id."),
}


def full_name(catalog: str, schema: str, table: str) -> str:
    return f"`{catalog}`.`{schema}`.`{table}`"


def ensure_tables(spark: SparkSession, catalog: str, schema: str) -> None:
    for table, (columns, comment) in _TABLES.items():
        spark.sql(
            f"CREATE TABLE IF NOT EXISTS {full_name(catalog, schema, table)} ({columns}) COMMENT '{comment}'"
        )


def merge(spark: SparkSession, source: DataFrame, target: str, keys: list[str], update_when: str | None = None) -> None:
    """Insert rows whose keys are new; update existing rows only when `update_when` (SQL on t./s.) holds."""
    view = f"merge_source_{uuid.uuid4().hex}"
    source.createOrReplaceTempView(view)
    on = " AND ".join(f"t.{k} = s.{k}" for k in keys)
    statement = f"MERGE INTO {target} t USING {view} s ON {on}"
    if update_when:
        statement += f" WHEN MATCHED AND ({update_when}) THEN UPDATE SET *"
    statement += " WHEN NOT MATCHED THEN INSERT *"
    spark.sql(statement)
    spark.catalog.dropTempView(view)
