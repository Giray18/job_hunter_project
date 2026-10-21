"""Translate a job posting to English and extract the fields used for matching."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from job_hunter import countries
from job_hunter.records import JobPosting, Seniority, SilverJob, WorkMode

MAX_SKILLS = 25


class EnrichedJob(BaseModel):
    """LLM output for one posting; validated before anything is stored."""

    model_config = ConfigDict(extra="forbid")

    language: str = Field(pattern=r"^[a-z]{2}$", description="ISO 639-1 code of the original description, e.g. en")
    title_en: str = Field(min_length=1)
    description_en: str = Field(description="English translation, or empty string if already English")
    summary: str = Field(min_length=1)
    skills: list[str] = Field(max_length=MAX_SKILLS)
    seniority: Seniority
    work_mode: WorkMode
    country_code: str | None = Field(pattern=r"^[A-Z]{2}$", description="ISO 3166-1 alpha-2 code, e.g. HR")
    city: str | None
    min_years_experience: int | None = Field(ge=0, le=50)

    @field_validator("language", mode="before")
    @classmethod
    def _lower_language(cls, v):
        return v.strip().lower() if isinstance(v, str) else v

    @field_validator("skills", mode="before")
    @classmethod
    def _dedupe_skills(cls, v):
        cleaned = [s.strip() for s in v or [] if isinstance(s, str) and s.strip()]
        return list(dict.fromkeys(cleaned))[:MAX_SKILLS]

    @field_validator("country_code", mode="before")
    @classmethod
    def _iso_country(cls, v):
        return countries.normalize_code(v) if isinstance(v, str) and v.strip() else None

    @field_validator("city", mode="before")
    @classmethod
    def _blank_city(cls, v):
        return v.strip() or None if isinstance(v, str) else v


SYSTEM_PROMPT = """You normalize job postings for a job-matching system.
Rules:
- language: ISO 639-1 code of the description's language.
- description_en: if the description is already in English, return an empty string "".
  Otherwise a faithful, complete English translation; keep paragraphs and bullet points; do not summarize,
  add or omit requirements.
- title_en: the job title in English.
- summary: 2-3 sentences on the role, main responsibilities and must-have requirements.
- skills: concrete technologies, tools, methods and domain skills mentioned (e.g. "Python", "Databricks", "SQL"),
  deduplicated, in English, at most 25.
- seniority: infer from title, criteria and required experience.
- work_mode: "remote" / "hybrid" / "onsite" only if the posting says so; otherwise "unknown".
- country_code: ISO 3166-1 alpha-2 code of the job's country (e.g. "HR" for Croatia, "GB" for the United Kingdom),
  from the location, or from the description if the location lacks it; null if not stated anywhere.
- city: in English (e.g. "Vienna"), from the location or description; null if not stated.
- min_years_experience: minimum years required if stated, else null.
Use only information in the posting. Never invent facts."""


def user_prompt(posting: JobPosting) -> str:
    criteria = "\n".join(f"- {k}: {v}" for k, v in (posting.criteria or {}).items())
    return (
        f"Title: {posting.title}\n"
        f"Company: {posting.company}\n"
        f"Location: {posting.location}\n"
        f"Criteria:\n{criteria or '- none'}\n\n"
        f"Description:\n{posting.description}"
    )


def to_silver_job(posting: JobPosting, result: EnrichedJob, model: str, enriched_at: datetime) -> SilverJob:
    """English postings come back with an empty description_en (saves output tokens); keep the original then."""
    return SilverJob(
        job_id=posting.job_id,
        language=result.language,
        title_en=result.title_en,
        description_en=result.description_en.strip() or (posting.description or ""),
        summary=result.summary,
        skills=result.skills,
        seniority=result.seniority,
        work_mode=result.work_mode,
        country=countries.name(result.country_code) if result.country_code else None,
        country_code=result.country_code,
        city=result.city,
        min_years_experience=result.min_years_experience,
        model=model,
        source_fetched_at=posting.fetched_at,
        enriched_at=enriched_at,
    )
