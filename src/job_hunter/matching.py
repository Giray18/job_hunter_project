"""Job ↔ CV matching: candidate filter, similarity shortlist (SQL) and LLM fit judging.

The candidate filter exists twice on purpose: as SQL (what runs on Databricks) and as a Python mirror.
Unit tests run both on the same cases (the SQL on DuckDB) and require identical results.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

from pydantic import BaseModel, ConfigDict, Field, field_validator

from job_hunter import countries
from job_hunter.cv import CvDocument
from job_hunter.records import Priority, RemoteEligibility, Seniority, WorkMode

# Remote-eligible area: EU + EEA (IS, LI, NO) + UK + CH + Western Balkans.
DEFAULT_REMOTE_COUNTRIES = (
    "AT,BE,BG,HR,CY,CZ,DK,EE,FI,FR,DE,GR,HU,IE,IT,LV,LT,LU,MT,NL,PL,PT,RO,SK,SI,ES,SE,"
    "IS,LI,NO,GB,CH,RS,BA,ME,MK,AL,XK"
)
DEFAULT_ALLOWED_SENIORITY = "mid,senior,lead,principal,unknown"
# Work modes a job outside the target country may have; "unknown" is resolved by the judge.
REMOTE_CANDIDATE_WORK_MODES = (WorkMode.REMOTE, WorkMode.UNKNOWN)


@dataclass(frozen=True)
class MatchConfig:
    target_country: str = "HR"
    remote_countries: tuple[str, ...] = tuple(DEFAULT_REMOTE_COUNTRIES.split(","))
    allowed_seniority: tuple[Seniority, ...] = tuple(Seniority(s) for s in DEFAULT_ALLOWED_SENIORITY.split(","))
    max_age_days: int = 14
    max_judge: int = 20
    croatia_min_fit: int = 60

    @classmethod
    def from_strings(cls, target_country: str, remote_countries: str, allowed_seniority: str, max_age_days: int,
                     max_judge: int, croatia_min_fit: int) -> "MatchConfig":
        """Parse bundle-variable strings; invalid country codes or seniority values raise ValueError."""
        if max_age_days <= 0 or max_judge <= 0 or not 0 <= croatia_min_fit <= 100:
            raise ValueError("max_age_days and max_judge must be positive; croatia_min_fit must be 0-100")
        return cls(
            target_country=countries.normalize_code(target_country),
            remote_countries=tuple(countries.normalize_code(c) for c in remote_countries.split(",") if c.strip()),
            allowed_seniority=tuple(Seniority(s.strip()) for s in allowed_seniority.split(",") if s.strip()),
            max_age_days=max_age_days,
            max_judge=max_judge,
            croatia_min_fit=croatia_min_fit,
        )

    def min_first_seen(self, now: datetime) -> datetime:
        return now - timedelta(days=self.max_age_days)


# ---------------------------------------------------------------- matching text


def match_text(title_en: str, seniority: str, work_mode: str, country: str | None, skills: list[str],
               summary: str) -> str:
    """Short text embedded per job (role and skills, not description boilerplate)."""
    return f"{title_en} | {seniority} | {work_mode} | {country or 'unknown country'}\nSkills: {', '.join(skills)}\n{summary}"


# ---------------------------------------------------------------- candidate filter


def passes_filter(country_code: str | None, work_mode: str, seniority: str, first_seen_at: datetime | None,
                  config: MatchConfig, now: datetime) -> bool:
    """Python mirror of `candidate_filter_sql` (kept identical by tests)."""
    if first_seen_at is None or first_seen_at < config.min_first_seen(now):
        return False
    if seniority not in config.allowed_seniority:
        return False
    if country_code == config.target_country:
        return True
    in_area = country_code is None or country_code in config.remote_countries
    return in_area and work_mode in REMOTE_CANDIDATE_WORK_MODES


def candidate_filter_sql(alias: str = "e") -> str:
    """WHERE clause over silver_job_embeddings columns; values come only from named parameters."""
    return f"""(
        {alias}.first_seen_at >= :min_first_seen
        AND array_contains(split(:allowed_seniority, ','), {alias}.seniority)
        AND (
            {alias}.country_code = :target_country
            OR (
                ({alias}.country_code IS NULL OR array_contains(split(:remote_countries, ','), {alias}.country_code))
                AND array_contains(split(:remote_work_modes, ','), {alias}.work_mode)
            )
        )
    )"""


def filter_params(config: MatchConfig, now: datetime) -> dict:
    return {
        "min_first_seen": config.min_first_seen(now),
        "allowed_seniority": ",".join(config.allowed_seniority),
        "target_country": config.target_country,
        "remote_countries": ",".join(config.remote_countries),
        "remote_work_modes": ",".join(REMOTE_CANDIDATE_WORK_MODES),
    }


def shortlist_sql(embeddings: str, silver_jobs: str, silver_cv: str, matches: str, max_judge: int) -> str:
    """Unjudged candidates for the current CV, target country first, then by cosine similarity."""
    if not isinstance(max_judge, int) or max_judge <= 0:
        raise ValueError("max_judge must be a positive int")
    return f"""
        SELECT e.job_id, e.title_en, e.company, e.url, e.country_code, e.work_mode, e.seniority, e.first_seen_at,
               s.country, s.city, s.min_years_experience, s.description_en,
               vector_cosine_similarity(e.embedding, cv.embedding) AS similarity
        FROM {embeddings} e
        JOIN {silver_jobs} s ON s.job_id = e.job_id
        CROSS JOIN (SELECT embedding FROM {silver_cv} WHERE cv_version = :cv_version) cv
        WHERE {candidate_filter_sql("e")}
          AND NOT EXISTS (SELECT 1 FROM {matches} m WHERE m.job_id = e.job_id AND m.cv_version = :cv_version)
        ORDER BY COALESCE(e.country_code = :target_country, false) DESC, similarity DESC
        LIMIT {max_judge}
    """


# ---------------------------------------------------------------- LLM judge

MAX_SKILL_LIST = 15


def _dedupe(values, limit: int) -> list[str]:
    cleaned = [v.strip() for v in values or [] if isinstance(v, str) and v.strip()]
    return list(dict.fromkeys(cleaned))[:limit]


class JobFit(BaseModel):
    """LLM judgement of how well the candidate fits one job."""

    model_config = ConfigDict(extra="forbid")

    fit_score: int = Field(ge=0, le=100)
    remote_eligibility: RemoteEligibility
    reasons: str = Field(min_length=1, max_length=1500)
    matching_skills: list[str] = Field(max_length=MAX_SKILL_LIST)
    missing_skills: list[str] = Field(max_length=MAX_SKILL_LIST)

    @field_validator("matching_skills", "missing_skills", mode="before")
    @classmethod
    def _cap_skills(cls, v):
        return _dedupe(v, MAX_SKILL_LIST)


JUDGE_SYSTEM_PROMPT = """You assess how well a candidate fits a job, for a personal job search.
Score fit_score from 0 to 100 using only the CV and the job posting:
- 90-100: near-perfect match; meets all must-haves and most nice-to-haves.
- 70-89: strong match with minor gaps.
- 50-69: partial match; some must-haves missing.
- 0-49: weak match or a different profession.
Weigh must-have requirements, core technologies, seniority and years of experience, and domain.
remote_eligibility (the candidate lives in Zagreb, Croatia and holds a Croatian residence and work permit):
- Job located in Croatia: "yes" if it is remote or hybrid, "no" if it requires being onsite every day,
  "unclear" if the posting does not say.
- Job located abroad or with no stated country: "yes" only if the posting says the job is fully remote and open to
  someone living in Croatia (or the EU/Europe); "no" if it clearly requires onsite or hybrid work abroad,
  relocation, or residency/tax residency in another country; "unclear" if the posting does not say.
reasons: 2-4 sentences explaining the score and the remote decision.
matching_skills / missing_skills: the job's key requirements the candidate has / lacks (max 15 each).
Never assume skills or experience that are not in the CV."""


def candidate_profile(cv: CvDocument) -> str:
    """Compact CV text for the judge (no contact details)."""
    lines = [
        f"Headline: {cv.headline}",
        f"Location: {cv.location}",
        f"Work authorization: {cv.work_authorization or 'not stated'}",
        f"Summary: {cv.summary}",
        "Skills: " + "; ".join(f"{g.category}: {', '.join(g.items)}" for g in cv.skills),
        "Certifications: " + "; ".join(cv.certifications),
        "Languages: " + ", ".join(f"{lang.language} ({lang.level})" for lang in cv.languages),
        "Experience:",
    ]
    for job in cv.experience:
        lines.append(f"- {job.role} at {job.company} ({job.start} - {job.end})")
        lines.extend(f"  * {b}" for b in job.bullets)
    lines.append("Education: " + "; ".join(f"{e.degree}, {e.institution}" for e in cv.education))
    return "\n".join(lines)


def job_description(job: dict) -> str:
    location = ", ".join(p for p in (job.get("city"), job.get("country")) if p) or "not stated"
    return (
        f"Title: {job['title_en']}\n"
        f"Company: {job.get('company')}\n"
        f"Location: {location}\n"
        f"Work mode: {job['work_mode']}\n"
        f"Seniority: {job['seniority']}\n"
        f"Minimum years of experience: {job.get('min_years_experience') or 'not stated'}\n\n"
        f"Description:\n{job['description_en']}"
    )


def judge_prompt(profile: str, job: dict) -> str:
    return f"CANDIDATE CV\n{profile}\n\nJOB POSTING\n{job_description(job)}"


def remote_eligibility(country_code: str | None, work_mode: str, judged: RemoteEligibility,
                       config: MatchConfig) -> RemoteEligibility:
    """For Croatian jobs with a known work mode the rule is fixed (remote/hybrid → yes, onsite → no);
    everything else uses the judge's answer."""
    if country_code == config.target_country:
        if work_mode in (WorkMode.REMOTE, WorkMode.HYBRID):
            return RemoteEligibility.YES
        if work_mode == WorkMode.ONSITE:
            return RemoteEligibility.NO
    return judged


def final_priority(country_code: str | None, eligibility: RemoteEligibility, config: MatchConfig) -> Priority:
    """Croatian jobs always keep priority; jobs abroad depend on remote eligibility."""
    if country_code == config.target_country:
        return Priority.CROATIA
    return {
        RemoteEligibility.YES: Priority.REMOTE,
        RemoteEligibility.UNCLEAR: Priority.REMOTE_UNCLEAR,
        RemoteEligibility.NO: Priority.EXCLUDED,
    }[eligibility]
