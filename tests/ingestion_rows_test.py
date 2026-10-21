"""Typed records ↔ table schemas: field order, row conversion and additive migrations."""

from dataclasses import fields
from datetime import UTC, datetime

import pytest

from job_hunter import storage
from job_hunter.records import AlertEmail, ApplyType, JobCard, JobPosting, Seniority, SilverJob, WorkMode, to_row

NOW = datetime(2026, 10, 20, 8, 0, tzinfo=UTC)


@pytest.mark.parametrize("table", list(storage.Table))
def test_record_fields_match_table_columns(table):
    spec = storage.TABLES[table]
    assert [f.name for f in fields(spec.record)] == [name for name, _ in storage.columns(spec.ddl)]


def test_columns_keeps_commas_inside_types():
    assert storage.columns("a STRING NOT NULL, b MAP<STRING, STRING>, c ARRAY<STRUCT<x: INT, y: INT>>") == [
        ("a", "STRING"),
        ("b", "MAP<STRING, STRING>"),
        ("c", "ARRAY<STRUCT<x: INT, y: INT>>"),
    ]


def test_to_row_flattens_nested_cards():
    email = AlertEmail(
        message_id="m1",
        thread_id="t1",
        received_at=NOW,
        sender="jobalerts-noreply@linkedin.com",
        subject="Data Engineer",
        snippet="snippet",
        text="text",
        job_cards=[JobCard(job_id="1", url="u", title="DE", company="Acme", location="Graz")],
        ingested_at=NOW,
    )
    assert to_row(email) == (
        "m1", "t1", NOW, "jobalerts-noreply@linkedin.com", "Data Engineer", "snippet", "text",
        [("1", "u", "DE", "Acme", "Graz")], NOW,
    )


def test_to_row_failed_posting_has_nulls():
    posting = JobPosting(
        job_id="1", url="u", title=None, company=None, location=None, description=None, criteria=None,
        apply_type=ApplyType.UNKNOWN, apply_checked_at=None, http_status=404, fetched_at=NOW,
    )
    assert to_row(posting) == ("1", "u", None, None, None, None, None, "unknown", None, 404, NOW)
    assert not posting.ok


def test_to_row_stores_enum_values():
    job = SilverJob(
        job_id="1", language="en", title_en="DE", description_en="d", summary="s", skills=["SQL"],
        seniority=Seniority.SENIOR, work_mode=WorkMode.REMOTE, country="Croatia", country_code="HR", city="Zagreb",
        min_years_experience=3, model="m", source_fetched_at=NOW, enriched_at=NOW,
    )
    row = to_row(job)
    assert row[6] == "senior" and type(row[6]) is str
    assert row[7] == "remote" and type(row[7]) is str
