"""Medallion table definitions, additive schema migration and an idempotent MERGE helper."""

import uuid
from dataclasses import dataclass, field
from enum import StrEnum

from pyspark.sql import DataFrame, SparkSession

from job_hunter import records


class Table(StrEnum):
    EMAILS = "bronze_job_alert_emails"
    POSTINGS = "bronze_job_postings"
    SILVER_JOBS = "silver_jobs"
    SILVER_CV = "silver_cv"
    SILVER_JOB_EMBEDDINGS = "silver_job_embeddings"
    GOLD_JOB_MATCHES = "gold_job_matches"


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

SILVER_JOBS_SCHEMA = """
    job_id STRING NOT NULL,
    language STRING,
    title_en STRING,
    description_en STRING,
    summary STRING,
    skills ARRAY<STRING>,
    seniority STRING,
    work_mode STRING,
    country STRING,
    country_code STRING,
    city STRING,
    min_years_experience INT,
    model STRING,
    source_fetched_at TIMESTAMP,
    enriched_at TIMESTAMP
"""

SILVER_CV_SCHEMA = """
    cv_version STRING NOT NULL,
    source_path STRING,
    cv_text STRING,
    cv_json STRING,
    match_text STRING,
    embedding ARRAY<FLOAT>,
    model STRING,
    embedding_model STRING,
    parsed_at TIMESTAMP,
    is_current BOOLEAN
"""


SILVER_JOB_EMBEDDINGS_SCHEMA = """
    job_id STRING NOT NULL,
    match_text STRING,
    embedding ARRAY<FLOAT>,
    embedding_model STRING,
    title_en STRING,
    company STRING,
    url STRING,
    country_code STRING,
    work_mode STRING,
    seniority STRING,
    first_seen_at TIMESTAMP,
    embedded_at TIMESTAMP
"""

GOLD_JOB_MATCHES_SCHEMA = """
    job_id STRING NOT NULL,
    cv_version STRING NOT NULL,
    similarity DOUBLE,
    priority STRING,
    fit_score INT,
    remote_eligibility STRING,
    reasons STRING,
    matching_skills ARRAY<STRING>,
    missing_skills ARRAY<STRING>,
    model STRING,
    judged_at TIMESTAMP
"""


@dataclass(frozen=True)
class TableSpec:
    ddl: str
    record: type
    comment: str
    properties: dict[str, str] = field(default_factory=dict)


TABLES: dict[Table, TableSpec] = {
    Table.EMAILS: TableSpec(
        EMAILS_SCHEMA, records.AlertEmail, "LinkedIn job-alert emails from Gmail (URL query strings stripped)."
    ),
    Table.POSTINGS: TableSpec(
        POSTINGS_SCHEMA, records.JobPosting, "LinkedIn job postings ('About the job' text and criteria) by job id."
    ),
    Table.SILVER_JOBS: TableSpec(
        SILVER_JOBS_SCHEMA, records.SilverJob, "Job postings translated to English with LLM-extracted matching fields."
    ),
    Table.SILVER_CV: TableSpec(
        SILVER_CV_SCHEMA, records.CvVersion, "Parsed CV versions: structured JSON, matching text and embedding."
    ),
    Table.SILVER_JOB_EMBEDDINGS: TableSpec(
        SILVER_JOB_EMBEDDINGS_SCHEMA,
        records.JobEmbedding,
        "One GTE embedding of match_text per job, plus filter columns. Source for the Vector Search index.",
        # Delta Sync Vector Search indexes require Change Data Feed on the source table.
        properties={"delta.enableChangeDataFeed": "true"},
    ),
    Table.GOLD_JOB_MATCHES: TableSpec(
        GOLD_JOB_MATCHES_SCHEMA, records.JobMatch, "LLM-judged job fit per CV version, with location priority."
    ),
}


def full_name(catalog: str, schema: str, table: Table) -> str:
    return f"`{catalog}`.`{schema}`.`{table.value}`"


def columns(ddl: str) -> list[tuple[str, str]]:
    """(name, type) pairs of a column DDL string; commas inside <...> belong to the type."""
    parts, depth, current = [], 0, ""
    for ch in ddl:
        depth += {"<": 1, ">": -1}.get(ch, 0)
        if ch == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += ch
    parts.append(current)
    result = []
    for part in (p.strip() for p in parts):
        if part:
            name, col_type = part.split(None, 1)
            result.append((name, col_type.removesuffix("NOT NULL").strip()))
    return result


def ensure_tables(spark: SparkSession, catalog: str, schema: str) -> None:
    """Create missing tables and add new columns to existing ones (additive migrations only)."""
    for table, spec in TABLES.items():
        target = full_name(catalog, schema, table)
        escaped = spec.comment.replace("\\", "\\\\").replace("'", "\\'")
        spark.sql(f"CREATE TABLE IF NOT EXISTS {target} ({spec.ddl}) COMMENT '{escaped}'")
        if spec.properties:
            props = ", ".join(f"'{k}' = '{v}'" for k, v in spec.properties.items())
            spark.sql(f"ALTER TABLE {target} SET TBLPROPERTIES ({props})")

        existing = set(spark.table(target).columns)
        previous = None
        for name, col_type in columns(spec.ddl):
            if name not in existing:
                position = f" AFTER {previous}" if previous else " FIRST"
                spark.sql(f"ALTER TABLE {target} ADD COLUMNS ({name} {col_type}{position})")
                print(f"Added column {name} to {target}")
            previous = name


def to_dataframe(spark: SparkSession, table: Table, rows: list) -> DataFrame:
    """Typed records → DataFrame with the table's schema."""
    return spark.createDataFrame([records.to_row(r) for r in rows], TABLES[table].ddl)


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


def update_columns(spark: SparkSession, source: DataFrame, target: str, keys: list[str], columns: list[str]) -> None:
    """Update only `columns` of existing rows matched on `keys` (no inserts)."""
    view = f"update_source_{uuid.uuid4().hex}"
    source.createOrReplaceTempView(view)
    on = " AND ".join(f"t.{k} = s.{k}" for k in keys)
    assignments = ", ".join(f"t.{c} = s.{c}" for c in columns)
    spark.sql(f"MERGE INTO {target} t USING {view} s ON {on} WHEN MATCHED THEN UPDATE SET {assignments}")
    spark.catalog.dropTempView(view)
