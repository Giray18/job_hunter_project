"""Translate a job posting to English and extract the fields used for matching."""

from datetime import datetime

SENIORITY = ["intern", "junior", "mid", "senior", "lead", "principal", "unknown"]
WORK_MODES = ["remote", "hybrid", "onsite", "unknown"]

SCHEMA = {
    "type": "object",
    "properties": {
        "language": {"type": "string", "description": "ISO 639-1 code of the original description, e.g. en, nl, de"},
        "title_en": {"type": "string"},
        "description_en": {"type": "string"},
        "summary": {"type": "string"},
        "skills": {"type": "array", "items": {"type": "string"}},
        "seniority": {"type": "string", "enum": SENIORITY},
        "work_mode": {"type": "string", "enum": WORK_MODES},
        "country": {"type": ["string", "null"]},
        "city": {"type": ["string", "null"]},
        "min_years_experience": {"type": ["integer", "null"]},
    },
    "required": [
        "language", "title_en", "description_en", "summary", "skills", "seniority", "work_mode",
        "country", "city", "min_years_experience",
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You normalize job postings for a job-matching system.
Rules:
- description_en: if the description is already in English, return an empty string "".
  Otherwise a faithful, complete English translation; keep paragraphs and bullet points; do not summarize,
  add or omit requirements.
- title_en: the job title in English.
- summary: 2-3 sentences on the role, main responsibilities and must-have requirements.
- skills: concrete technologies, tools, methods and domain skills mentioned (e.g. "Python", "Databricks", "SQL"),
  deduplicated, in English, at most 25.
- seniority: infer from title, criteria and required experience.
- work_mode: "remote" / "hybrid" / "onsite" only if the posting says so; otherwise "unknown".
- country and city: from the location, or from the description if the location lacks them; in English
  (e.g. "Austria", "Vienna"); null if not stated anywhere.
- min_years_experience: minimum years required if stated, else null.
Use only information in the posting. Never invent facts."""


def user_prompt(posting: dict) -> str:
    criteria = "\n".join(f"- {k}: {v}" for k, v in (posting.get("criteria") or {}).items())
    return (
        f"Title: {posting.get('title')}\n"
        f"Company: {posting.get('company')}\n"
        f"Location: {posting.get('location')}\n"
        f"Criteria:\n{criteria or '- none'}\n\n"
        f"Description:\n{posting.get('description')}"
    )


def enriched_row(job_id: str, result: dict, original_description: str, model: str, source_fetched_at: datetime,
                 enriched_at: datetime) -> tuple:
    """LLM result → row in SILVER_JOBS_SCHEMA column order, with enums and lists sanitized.

    English postings come back with an empty description_en (saves output tokens); keep the original then.
    """
    skills = list(dict.fromkeys(s.strip() for s in result.get("skills") or [] if s and s.strip()))[:25]
    return (
        job_id,
        (result.get("language") or "").lower()[:5] or None,
        result.get("title_en"),
        (result.get("description_en") or "").strip() or original_description,
        result.get("summary"),
        skills,
        _enum(result.get("seniority"), SENIORITY),
        _enum(result.get("work_mode"), WORK_MODES),
        result.get("country") or None,
        result.get("city") or None,
        result.get("min_years_experience"),
        model,
        source_fetched_at,
        enriched_at,
    )


def _enum(value, allowed: list[str]) -> str:
    value = (value or "").strip().lower()
    return value if value in allowed else "unknown"
