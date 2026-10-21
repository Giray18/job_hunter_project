"""Tailored CV and cover letter per job, with fact checks against the source CV.

The LLM may only select, reorder and rephrase what is in the source CV. Validators (run with the source CV
as Pydantic validation context) reject invented employers, roles, dates, skills, certifications and numbers;
`llm.chat_json` feeds the errors back to the model and retries.
"""

import re
import unicodedata
from dataclasses import dataclass
from datetime import date

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

from job_hunter.cv import CvDocument, SkillGroup

MAX_BULLETS_PER_ROLE = 6
SUMMARY_MAX_WORDS = 80
LETTER_MIN_WORDS = 250
LETTER_MAX_WORDS = 350

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")

WORK_PERMIT_SENTENCE = (
    "I am based in Zagreb with a valid Croatian residence and work permit, so no sponsorship is needed."
)
DEFAULT_AVAILABILITY_NOTE = "I am available to start quickly"
DEFAULT_B2B_NOTE = "I am also open to B2B cooperation through my own company"


def normalize(text: str | None) -> str:
    """Comparison key: case-folded, accents/dash variants unified, whitespace collapsed."""
    text = unicodedata.normalize("NFKC", text or "").casefold()
    text = re.sub(r"[‐-―−]", "-", text)
    return re.sub(r"\s+", " ", text).strip()


def numbers(text: str) -> set[str]:
    return {n.replace(",", ".") for n in _NUMBER_RE.findall(text or "")}


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'’-]+\b", text))


def _source(info: ValidationInfo) -> CvDocument | None:
    return (info.context or {}).get("source")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- tailored CV


class TailoredExperience(_Strict):
    company: str
    role: str
    start: str
    end: str
    bullets: list[str] = Field(max_length=MAX_BULLETS_PER_ROLE)


class TailoredCv(_Strict):
    summary: str = Field(min_length=1)
    skills: list[SkillGroup]
    experience: list[TailoredExperience]
    certifications: list[str]

    @model_validator(mode="after")
    def _facts(self, info: ValidationInfo) -> "TailoredCv":
        source = _source(info)
        if source is None:
            return self
        problems = check_cv_facts(self, source)
        if problems:
            raise ValueError("; ".join(problems))
        return self


def check_cv_facts(tailored: TailoredCv, source: CvDocument) -> list[str]:
    problems = []
    if word_count(tailored.summary) > SUMMARY_MAX_WORDS:
        problems.append(f"summary has {word_count(tailored.summary)} words, max {SUMMARY_MAX_WORDS}")

    if len(tailored.experience) != len(source.experience):
        problems.append(f"experience must list all {len(source.experience)} roles in the original order, "
                        f"got {len(tailored.experience)}")
    for i, (new, old) in enumerate(zip(tailored.experience, source.experience, strict=False)):
        for field in ("company", "role", "start", "end"):
            if normalize(getattr(new, field)) != normalize(getattr(old, field)):
                problems.append(f"experience[{i}].{field} must be exactly {getattr(old, field)!r}, "
                                f"got {getattr(new, field)!r}")

    known_skills = {normalize(item) for group in source.skills for item in group.items}
    invented = [item for group in tailored.skills for item in group.items if normalize(item) not in known_skills]
    if invented:
        problems.append(f"skills not in the original CV: {invented}")

    known_certs = {normalize(c) for c in source.certifications}
    invented_certs = [c for c in tailored.certifications if normalize(c) not in known_certs]
    if invented_certs:
        problems.append(f"certifications not in the original CV: {invented_certs}")

    source_numbers = numbers(source.model_dump_json())
    rewritten = " ".join([tailored.summary] + [b for job in tailored.experience for b in job.bullets])
    invented_numbers = sorted(numbers(rewritten) - source_numbers)
    if invented_numbers:
        problems.append(f"numbers not in the original CV: {invented_numbers}")
    return problems


TAILOR_SYSTEM_PROMPT = f"""You tailor a candidate's CV to one job posting. Output JSON only.
You may only SELECT, REORDER and REPHRASE content from the source CV. Never invent anything.
- summary: rewrite the profile for this job, at most {SUMMARY_MAX_WORDS} words, first person implied, no buzzword lists.
- skills: the source CV's skill groups and items, reordered so the job's key requirements come first.
  Use only items that appear in the source CV (you may drop irrelevant items).
- experience: ALL roles of the source CV, in the same order, with company, role, start and end copied EXACTLY.
  bullets: choose and rephrase the source bullets that best match the job, most relevant first.
  The 3 most recent roles: 3-{MAX_BULLETS_PER_ROLE} bullets each. Other data/BI roles: 1-2 bullets.
  Non-data roles (sales, engineering, military): 0-1 bullet.
  Keep facts, tools and numbers exactly as in the source; do not add metrics.
- certifications: source certifications only, most relevant to the job first (you may drop irrelevant ones).
The CV must fit on two pages."""


def tailor_prompt(source: CvDocument, job: dict) -> str:
    cv_json = source.model_dump_json(exclude={"trainings", "projects", "email", "phone", "links"}, indent=1)
    return f"SOURCE CV (JSON)\n{cv_json}\n\nJOB POSTING\n{_job_text(job)}"


# ---------------------------------------------------------------- cover letter


class CoverLetter(_Strict):
    greeting: str = Field(min_length=1, max_length=120)
    paragraphs: list[str] = Field(min_length=2, max_length=3)

    @model_validator(mode="after")
    def _facts(self, info: ValidationInfo) -> "CoverLetter":
        context = info.context or {}
        source = context.get("source")
        if source is None:
            return self
        problems = check_letter_facts(self, source, context["job"], context["closing"])
        if problems:
            raise ValueError("; ".join(problems))
        return self


def check_letter_facts(letter: CoverLetter, source: CvDocument, job: dict, closing: str) -> list[str]:
    problems = []
    body = " ".join(letter.paragraphs)
    total = word_count(f"{letter.greeting} {body} {closing}")
    if not LETTER_MIN_WORDS <= total <= LETTER_MAX_WORDS:
        problems.append(f"letter has {total} words including the fixed closing paragraph "
                        f"({word_count(closing)} words); it must be {LETTER_MIN_WORDS}-{LETTER_MAX_WORDS}")
    company = job.get("company")
    if company and not company_mentioned(company, f"{letter.greeting} {body}"):
        problems.append(f"the letter must name the company {company!r}")
    allowed = numbers(source.model_dump_json()) | numbers(_job_text(job))
    invented = sorted(numbers(body) - allowed)
    if invented:
        problems.append(f"numbers not in the CV or the job posting: {invented}")
    return problems


# Words that don't identify a company on their own ("Teva Pharmaceuticals" is named by "Teva").
_GENERIC_COMPANY_WORDS = {
    "ab", "ag", "as", "bv", "co", "company", "consulting", "corp", "corporation", "d.o.o", "doo", "gmbh", "group",
    "holding", "inc", "international", "llc", "ltd", "nv", "oy", "pharmaceuticals", "plc", "sa", "services",
    "solutions", "systems", "technologies", "technology", "the",
}


def company_mentioned(company: str, text: str) -> bool:
    """True if the text names the company by its full name or by a distinctive word of it."""
    haystack = normalize(text)
    if normalize(company) in haystack:
        return True
    words = re.findall(r"[\w.]+", normalize(company))
    distinctive = [w.strip(".") for w in words if len(w.strip(".")) >= 3 and w.strip(".") not in _GENERIC_COMPANY_WORDS]
    return any(re.search(rf"\b{re.escape(w)}\b", haystack) for w in distinctive)


LETTER_SYSTEM_PROMPT =f"""You write the body of a cover letter for one job, in English. Output JSON only.
Tone: modern, direct, confident, specific; no clichés ("I am writing to express my interest"), no flattery.
- greeting: "Dear Hiring Team at <company>," unless the posting names a contact person.
- paragraphs: 2-3 paragraphs. 1) why this role and company, tied to what the posting says.
  2-3) two or three concrete achievements from the CV that match the job's key requirements, naming tools and results.
- Do NOT write a closing paragraph, sign-off, contact details, work permit or availability: they are added separately.
- Total length including the fixed closing paragraph given below: {LETTER_MIN_WORDS}-{LETTER_MAX_WORDS} words.
- Use only facts, employers, tools and numbers from the CV; never invent experience."""


def letter_prompt(source: CvDocument, tailored: TailoredCv, job: dict, closing: str) -> str:
    cv_json = source.model_dump_json(exclude={"trainings", "projects", "email", "phone", "links"}, indent=1)
    return (
        f"CANDIDATE CV (JSON)\n{cv_json}\n\nTAILORED SUMMARY\n{tailored.summary}\n\n"
        f"JOB POSTING\n{_job_text(job)}\n\nFIXED CLOSING PARAGRAPH (added after your paragraphs)\n{closing}"
    )


def closing_paragraph(availability_note: str = DEFAULT_AVAILABILITY_NOTE, b2b_note: str = DEFAULT_B2B_NOTE) -> str:
    return f"{WORK_PERMIT_SENTENCE} {availability_note.rstrip('.')}. {b2b_note.rstrip('.')}."


@dataclass(frozen=True)
class LetterDocument:
    """Final cover letter: fixed header and closing around the LLM-written greeting and body."""

    name: str
    contact_line: str
    date_line: str
    greeting: str
    paragraphs: list[str]
    closing: str
    sign_off: str

    def text(self) -> str:
        return "\n\n".join([self.name, self.contact_line, self.date_line, self.greeting, *self.paragraphs,
                            self.closing, self.sign_off])


def compose_letter(source: CvDocument, letter: CoverLetter, closing: str, today: date) -> LetterDocument:
    return LetterDocument(
        name=source.name,
        contact_line=contact_line(source),
        date_line=today.strftime("%d %B %Y"),
        greeting=letter.greeting,
        paragraphs=letter.paragraphs,
        closing=closing,
        sign_off=f"Kind regards,\n{source.name}",
    )


def contact_line(source: CvDocument) -> str:
    linkedin = next((link.url for link in source.links if "linkedin" in link.url.lower()), None)
    return " | ".join(p for p in (source.email, source.phone, source.location, linkedin) if p)


# ---------------------------------------------------------------- helpers


def _job_text(job: dict) -> str:
    location = ", ".join(p for p in (job.get("city"), job.get("country")) if p) or "not stated"
    return (
        f"Title: {job['title_en']}\nCompany: {job.get('company')}\nLocation: {location}\n"
        f"Key requirements the candidate matches: {', '.join(job.get('matching_skills') or [])}\n"
        f"Requirements the candidate lacks (do not claim them): {', '.join(job.get('missing_skills') or [])}\n\n"
        f"Description:\n{job['description_en']}"
    )
