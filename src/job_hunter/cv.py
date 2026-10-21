"""CV PDF → text → structured JSON (via LLM) → matching text for embeddings."""

import hashlib
import io
import json
import logging
import re
from datetime import datetime

from pypdf import PdfReader

_STR = {"type": "string"}
_NULLABLE_STR = {"type": ["string", "null"]}
_STR_LIST = {"type": "array", "items": _STR}


def _obj(properties: dict) -> dict:
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


SCHEMA = _obj(
    {
        "name": _STR,
        "headline": _STR,
        "summary": _STR,
        "location": _NULLABLE_STR,
        "work_authorization": _NULLABLE_STR,
        "email": _NULLABLE_STR,
        "phone": _NULLABLE_STR,
        "links": {"type": "array", "items": _obj({"label": _STR, "url": _STR})},
        "experience": {
            "type": "array",
            "items": _obj(
                {
                    "company": _STR,
                    "role": _STR,
                    "employment_type": _NULLABLE_STR,
                    "start": _STR,
                    "end": _STR,
                    "bullets": _STR_LIST,
                }
            ),
        },
        "skills": {"type": "array", "items": _obj({"category": _STR, "items": _STR_LIST})},
        "certifications": _STR_LIST,
        "education": {
            "type": "array",
            "items": _obj({"degree": _STR, "institution": _STR, "start": _STR, "end": _STR, "details": _STR_LIST}),
        },
        "languages": {"type": "array", "items": _obj({"language": _STR, "level": _STR})},
        "trainings": _STR_LIST,
        "projects": _STR_LIST,
    }
)

SYSTEM_PROMPT = """You convert the raw text of a CV into structured JSON.
The text was extracted from a multi-column PDF, so sections may appear out of order or interleaved; reassemble them.
Rules:
- Copy facts exactly: company names, roles, dates (as written, e.g. "Dec 2024", "Ongoing"), degrees, certifications.
- experience: every job, most recent first, with all its bullet points kept verbatim (only fix line breaks,
  drop bullet symbols like "•"). employment_type: e.g. "Contractor" when stated, else null.
- Page breaks can separate a job header from its bullets: a block of bullets with no job header (e.g. at the top
  of a page) belongs to the job whose header appears elsewhere without bullets (often at the end of that page).
  Every job should end up with its bullets.
- headline: the person's title line (e.g. "Data & AI Engineer"); summary: the profile statement, verbatim if present.
- skills: group the stack/skills section into categories as the CV does (or sensible ones like "AI", "Languages",
  "Platforms", "BI"), keeping the original wording.
- work_authorization: any statement about residence/work permit/sponsorship, verbatim, else null.
- trainings and projects: one string per item, verbatim.
Never invent, merge or drop items. Use null or [] when something is absent."""


def extract_text(pdf_bytes: bytes) -> str:
    # Icon fonts in designed CVs make pypdf log harmless "Odd-length string" warnings.
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()


def normalize(cv: dict) -> dict:
    """Strip leftover bullet symbols and whitespace from bullet points."""
    for job in cv.get("experience", []):
        job["bullets"] = [b for b in (_clean_bullet(b) for b in job.get("bullets", [])) if b]
    return cv


def roles_without_bullets(cv: dict) -> list[str]:
    return [f"{job['role']} at {job['company']}" for job in cv.get("experience", []) if not job.get("bullets")]


def _clean_bullet(text: str) -> str:
    return re.sub(r"^[\s•·▪●◦\-–*]+", "", text or "").strip()


def file_version(pdf_bytes: bytes) -> str:
    """Content hash identifying a CV version; unchanged files are not reprocessed."""
    return hashlib.sha256(pdf_bytes).hexdigest()[:16]


def match_text(cv: dict) -> str:
    """Short text embedded for job matching: headline, recent roles, skills and summary."""
    roles = "; ".join(f"{e['role']} at {e['company']}" for e in cv.get("experience", [])[:3])
    skills = ", ".join(item for group in cv.get("skills", []) for item in group.get("items", []))
    return (
        f"{cv.get('headline')} | {cv.get('location') or ''}\n"
        f"Recent roles: {roles}\n"
        f"Skills: {skills}\n"
        f"{cv.get('summary') or ''}"
    ).strip()


def cv_row(version: str, source_path: str, text: str, cv: dict, embedding: list[float], model: str,
           embedding_model: str, parsed_at: datetime) -> tuple:
    """Row in SILVER_CV_SCHEMA column order."""
    return (
        version,
        source_path,
        text,
        json.dumps(cv, ensure_ascii=False),
        match_text(cv),
        embedding,
        model,
        embedding_model,
        parsed_at,
        True,
    )
