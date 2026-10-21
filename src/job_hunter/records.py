"""Typed records passed between tasks: enums for fixed value sets and one dataclass per table row.

Each table-row dataclass lists its fields in the same order as the table's columns in `storage`;
a unit test enforces this, so `to_row()` always lines values up with the right columns.
"""

from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from enum import Enum, StrEnum
from http import HTTPStatus
from typing import Any


class Seniority(StrEnum):
    INTERN = "intern"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    PRINCIPAL = "principal"
    UNKNOWN = "unknown"


class WorkMode(StrEnum):
    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"
    UNKNOWN = "unknown"


# Fetch outcomes that are final: the posting was read, or LinkedIn removed it.
FINAL_HTTP_STATUSES = (HTTPStatus.OK, HTTPStatus.NOT_FOUND)


@dataclass(frozen=True)
class JobCard:
    """One job listed in a LinkedIn alert email."""

    job_id: str
    url: str
    title: str | None
    company: str | None
    location: str | None


@dataclass(frozen=True)
class AlertEmail:
    """Row of bronze_job_alert_emails."""

    message_id: str
    thread_id: str
    received_at: datetime | None
    sender: str | None
    subject: str | None
    snippet: str
    text: str
    job_cards: list[JobCard]
    ingested_at: datetime | None = None


@dataclass(frozen=True)
class JobPosting:
    """Row of bronze_job_postings."""

    job_id: str
    url: str
    title: str | None
    company: str | None
    location: str | None
    description: str | None
    criteria: dict[str, str] | None
    http_status: int
    fetched_at: datetime

    @property
    def ok(self) -> bool:
        return self.http_status == HTTPStatus.OK


@dataclass(frozen=True)
class SilverJob:
    """Row of silver_jobs."""

    job_id: str
    language: str
    title_en: str
    description_en: str
    summary: str
    skills: list[str]
    seniority: Seniority
    work_mode: WorkMode
    country: str | None
    country_code: str | None
    city: str | None
    min_years_experience: int | None
    model: str
    source_fetched_at: datetime
    enriched_at: datetime


@dataclass(frozen=True)
class CvVersion:
    """Row of silver_cv."""

    cv_version: str
    source_path: str
    cv_text: str
    cv_json: str
    match_text: str
    embedding: list[float]
    model: str
    embedding_model: str
    parsed_at: datetime
    is_current: bool


def to_row(record: Any) -> tuple:
    """Dataclass → tuple of plain values (enums → values, nested dataclasses → tuples) for createDataFrame."""
    return tuple(_plain(getattr(record, f.name)) for f in fields(record))


def _plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return to_row(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value
