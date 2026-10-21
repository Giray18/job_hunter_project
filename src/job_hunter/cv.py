"""CV PDF → text → structured, validated CvDocument (via LLM) → matching text for embeddings."""

import hashlib
import io
import logging
import re
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pypdf import PdfReader

from job_hunter.records import CvVersion


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Link(_Strict):
    label: str
    url: str


class Experience(_Strict):
    company: str = Field(min_length=1)
    role: str = Field(min_length=1)
    employment_type: str | None
    start: str = Field(min_length=1)
    end: str = Field(min_length=1)
    bullets: list[str]

    @field_validator("bullets", mode="before")
    @classmethod
    def _clean_bullets(cls, v):
        """Strip leftover bullet symbols and whitespace; drop empty bullets."""
        cleaned = (re.sub(r"^[\s•·▪●◦\-–*]+", "", b or "").strip() for b in v or [] if isinstance(b, str))
        return [b for b in cleaned if b]


class SkillGroup(_Strict):
    category: str
    items: list[str]


class Education(_Strict):
    degree: str = Field(min_length=1)
    institution: str = Field(min_length=1)
    start: str
    end: str
    details: list[str]


class Language(_Strict):
    language: str = Field(min_length=1)
    level: str


class CvDocument(_Strict):
    name: str = Field(min_length=1)
    headline: str = Field(min_length=1)
    summary: str
    location: str | None
    work_authorization: str | None
    email: str | None
    phone: str | None
    links: list[Link]
    experience: list[Experience] = Field(min_length=1)
    skills: list[SkillGroup]
    certifications: list[str]
    education: list[Education]
    languages: list[Language]
    trainings: list[str]
    projects: list[str]

    def roles_without_bullets(self) -> list[str]:
        return [f"{job.role} at {job.company}" for job in self.experience if not job.bullets]

    def match_text(self) -> str:
        """Short text embedded for job matching: headline, recent roles, skills and summary."""
        roles = "; ".join(f"{job.role} at {job.company}" for job in self.experience[:3])
        skills = ", ".join(item for group in self.skills for item in group.items)
        return (
            f"{self.headline} | {self.location or ''}\n"
            f"Recent roles: {roles}\n"
            f"Skills: {skills}\n"
            f"{self.summary}"
        ).strip()


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


def file_version(pdf_bytes: bytes) -> str:
    """Content hash identifying a CV version; unchanged files are not reprocessed."""
    return hashlib.sha256(pdf_bytes).hexdigest()[:16]


def to_cv_version(version: str, source_path: str, text: str, document: CvDocument, embedding: list[float],
                  model: str, embedding_model: str, parsed_at: datetime) -> CvVersion:
    return CvVersion(
        cv_version=version,
        source_path=source_path,
        cv_text=text,
        cv_json=document.model_dump_json(),
        match_text=document.match_text(),
        embedding=embedding,
        model=model,
        embedding_model=embedding_model,
        parsed_at=parsed_at,
        is_current=True,
    )
