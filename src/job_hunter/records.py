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


class Priority(StrEnum):
    """Location priority of a judged job."""

    CROATIA = "croatia"
    REMOTE = "remote"
    REMOTE_UNCLEAR = "remote_unclear"
    EXCLUDED = "excluded"


class RemoteEligibility(StrEnum):
    """Can the candidate (based in Zagreb) work this job without relocating?

    Croatia: yes for remote/hybrid, no for onsite-only. Abroad: yes only for fully remote work open to
    someone in Croatia; no if onsite/hybrid abroad or residency elsewhere is required; unclear if not stated.
    """

    YES = "yes"
    NO = "no"
    UNCLEAR = "unclear"


class ApplyType(StrEnum):
    """How a LinkedIn job is applied to (detected from the public job page)."""

    EXTERNAL = "external"  # "Apply / Başvur" → company's own application site (URL only shown when logged in)
    EASY_APPLY = "easy_apply"  # "Easy Apply / Kolay Başvuru" → form inside LinkedIn
    CLOSED = "closed"  # no apply button: LinkedIn no longer accepts applications
    UNKNOWN = "unknown"


class StatusValue(StrEnum):
    """Outcome of a job for the candidate; any status removes the job from future rankings."""

    APPLIED = "applied"
    SKIPPED = "skipped"  # the candidate decided not to apply
    CLOSED = "closed"  # no longer accepting applications
    NOT_ELIGIBLE = "not_eligible"  # excluded by the candidate's rules, e.g. onsite abroad


@dataclass(frozen=True)
class ApplicationStatus:
    """Row of gold_application_status."""

    job_id: str
    status: StatusValue
    source: str  # "pipeline", "browser" or "manual"
    notes: str | None
    updated_at: datetime


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
    apply_type: ApplyType
    apply_checked_at: datetime | None
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


@dataclass(frozen=True)
class JobEmbedding:
    """Row of silver_job_embeddings."""

    job_id: str
    match_text: str
    embedding: list[float]
    embedding_model: str
    title_en: str
    company: str | None
    url: str
    country_code: str | None
    work_mode: WorkMode
    seniority: Seniority
    first_seen_at: datetime | None
    embedded_at: datetime


@dataclass(frozen=True)
class JobMatch:
    """Row of gold_job_matches: one judged job for one CV version."""

    job_id: str
    cv_version: str
    similarity: float
    priority: Priority
    fit_score: int
    remote_eligibility: RemoteEligibility
    reasons: str
    matching_skills: list[str]
    missing_skills: list[str]
    model: str
    judged_at: datetime


@dataclass(frozen=True)
class Application:
    """Row of gold_applications: tailored CV + cover letter generated for one ranked job."""

    job_id: str
    cv_version: str
    rank_position: int
    tier: int
    fit_score: int
    tailored_cv_json: str
    cover_letter_text: str
    cv_pdf_path: str
    cover_letter_pdf_path: str
    model: str
    created_at: datetime


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
