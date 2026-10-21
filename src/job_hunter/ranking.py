"""Ranking of judged jobs; Milestone 5 tailors CVs for the top N."""

from dataclasses import dataclass

from job_hunter.records import Priority


@dataclass(frozen=True)
class RankedJob:
    job_id: str
    title_en: str
    company: str | None
    country_code: str | None
    url: str
    priority: Priority
    fit_score: int
    similarity: float
    reasons: str


def tier(job: RankedJob, priority_min_fit: int) -> int:
    """1 = Croatian job with fit >= priority_min_fit; 2 = confirmed remote job with fit >= priority_min_fit;
    3 = everything else (weaker Croatian/remote jobs and unclear-remote jobs), competing on fit score."""
    if job.fit_score >= priority_min_fit:
        if job.priority is Priority.CROATIA:
            return 1
        if job.priority is Priority.REMOTE:
            return 2
    return 3


def rank(jobs: list[RankedJob], priority_min_fit: int) -> list[RankedJob]:
    eligible = [j for j in jobs if j.priority is not Priority.EXCLUDED]
    return sorted(eligible, key=lambda j: (tier(j, priority_min_fit), -j.fit_score, -j.similarity))


def top_jobs(jobs: list[RankedJob], priority_min_fit: int, n: int = 10) -> list[RankedJob]:
    return rank(jobs, priority_min_fit)[:n]


def judged_jobs_sql(embeddings: str, matches: str, statuses: str) -> str:
    """Latest judgement per job seen within the age window (any CV version), excluding jobs that already have a
    status (applied, skipped, closed, not eligible)."""
    return f"""
        SELECT m.job_id, e.title_en, e.company, e.country_code, e.url, m.priority, m.fit_score, m.similarity,
               m.reasons
        FROM {matches} m
        JOIN {embeddings} e ON e.job_id = m.job_id
        LEFT ANTI JOIN {statuses} st ON st.job_id = m.job_id
        WHERE e.first_seen_at >= :min_first_seen
        QUALIFY row_number() OVER (PARTITION BY m.job_id ORDER BY m.judged_at DESC) = 1
    """


def from_row(row: dict) -> RankedJob:
    return RankedJob(
        job_id=row["job_id"],
        title_en=row["title_en"],
        company=row["company"],
        country_code=row["country_code"],
        url=row["url"],
        priority=Priority(row["priority"]),
        fit_score=row["fit_score"],
        similarity=row["similarity"],
        reasons=row["reasons"],
    )


def describe(jobs: list[RankedJob], priority_min_fit: int) -> str:
    lines = []
    for i, j in enumerate(jobs, 1):
        lines.append(
            f"{i}. [{j.priority} t{tier(j, priority_min_fit)}] fit {j.fit_score} sim {j.similarity:.3f} | "
            f"{j.title_en} | {j.company} | {j.country_code or '-'} | {j.url}"
        )
    return "\n".join(lines)
