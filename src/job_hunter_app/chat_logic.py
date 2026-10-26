"""Pure logic of the job chat app: filters, SQL, prompt and answer parsing (no Databricks calls, unit-tested)."""

import json
from dataclasses import dataclass, field

# Index columns returned by a vector search; the job details come from SQL afterwards.
SEARCH_COLUMNS = ["job_id", "title_en", "company"]
NO_STATUS = "none"

SYSTEM_PROMPT = """You help a job seeker explore job postings from their own daily pipeline.
Answer only from the jobs listed in the context; if they don't answer the question, say so.
Refer to jobs as [n] with company and title, e.g. "[2] Codat - Senior Data Engineer".
Mention the fit score (0-100, from an LLM judge against the CV) and the application status when they matter.
Be concise: a short answer, then a bullet per relevant job."""


@dataclass(frozen=True)
class Filters:
    """UI filters; empty lists mean 'any'."""

    country_codes: list[str] = field(default_factory=list)
    work_modes: list[str] = field(default_factory=list)
    seniorities: list[str] = field(default_factory=list)
    statuses: list[str] = field(default_factory=list)  # NO_STATUS = jobs without a status yet
    min_fit: int = 0
    max_age_days: int = 30


def index_filters(filters: Filters) -> dict[str, list[str]]:
    """Filters the vector index can apply itself (it only has these columns)."""
    out = {}
    if filters.country_codes:
        out["country_code"] = filters.country_codes
    if filters.work_modes:
        out["work_mode"] = filters.work_modes
    if filters.seniorities:
        out["seniority"] = filters.seniorities
    return out


def index_filters_json(filters: Filters) -> str | None:
    selected = index_filters(filters)
    return json.dumps(selected) if selected else None


def jobs_sql(catalog: str, schema: str, filters: Filters, job_ids: list[str] | None = None,
             limit: int = 200) -> tuple[str, dict]:
    """One row per job with match, application and status details; returns (sql, named parameters).

    With `job_ids` (search hits) only those jobs are returned; the caller keeps the search order.
    """
    t = f"`{catalog}`.`{schema}`"
    params: dict = {"min_fit": int(filters.min_fit), "max_age_days": int(filters.max_age_days)}
    where = [
        "e.first_seen_at >= current_timestamp - make_interval(0, 0, 0, :max_age_days)",
        "coalesce(m.fit_score, 0) >= :min_fit",
    ]
    for column, values, prefix in (("e.country_code", filters.country_codes, "cc"),
                                   ("e.work_mode", filters.work_modes, "wm"),
                                   ("e.seniority", filters.seniorities, "sn")):
        if values:
            where.append(f"{column} IN ({_placeholders(prefix, values, params)})")
    if filters.statuses:
        named = [s for s in filters.statuses if s != NO_STATUS]
        parts = [f"st.status IN ({_placeholders('st', named, params)})"] if named else []
        if NO_STATUS in filters.statuses:
            parts.append("st.status IS NULL")
        where.append("(" + " OR ".join(parts) + ")")
    if job_ids is not None:
        where.append(f"e.job_id IN ({_placeholders('id', job_ids, params)})" if job_ids else "FALSE")
    sql = f"""
        WITH m AS (
            SELECT * FROM {t}.gold_job_matches
            QUALIFY row_number() OVER (PARTITION BY job_id ORDER BY judged_at DESC) = 1
        ), a AS (
            SELECT * FROM {t}.gold_applications
            QUALIFY row_number() OVER (PARTITION BY job_id ORDER BY created_at DESC) = 1
        )
        SELECT e.job_id, e.title_en, e.company, e.url, e.country_code, s.city, e.work_mode, e.seniority,
               e.first_seen_at, s.summary, m.fit_score, m.priority, m.reasons, m.missing_skills,
               a.cv_pdf_path, a.cover_letter_pdf_path, st.status
        FROM {t}.silver_job_embeddings e
        LEFT JOIN {t}.silver_jobs s ON s.job_id = e.job_id
        LEFT JOIN m ON m.job_id = e.job_id
        LEFT JOIN a ON a.job_id = e.job_id
        LEFT JOIN {t}.gold_application_status st ON st.job_id = e.job_id
        WHERE {" AND ".join(where)}
        ORDER BY m.fit_score DESC NULLS LAST, e.first_seen_at DESC
        LIMIT {int(limit)}
    """
    return sql, params


def parse_hits(column_names: list[str], data_array: list[list] | None) -> list[dict]:
    """Index hits as dicts in rank order; the index appends its relevance `score` as the last column."""
    return [dict(zip(column_names, row, strict=True)) for row in data_array or []]


def order_like(rows: list[dict], job_ids: list[str]) -> list[dict]:
    """Rows in search-hit order (SQL returns them by fit)."""
    position = {job_id: i for i, job_id in enumerate(job_ids)}
    return sorted(rows, key=lambda r: position.get(r["job_id"], len(position)))


def with_hits(rows: list[dict], hits: list[dict]) -> list[dict]:
    """SQL rows in hit order, each with its search rank and index score."""
    by_id = {h["job_id"]: (rank, h.get("score")) for rank, h in enumerate(hits, 1)}
    ordered = order_like(rows, [h["job_id"] for h in hits])
    return [{**r, "search_rank": by_id[r["job_id"]][0], "search_score": by_id[r["job_id"]][1]}
            for r in ordered if r["job_id"] in by_id]


def build_context(rows: list[dict], max_summary_chars: int = 600) -> str:
    """Numbered job blocks for the LLM, in the given order."""
    blocks = []
    for n, r in enumerate(rows, 1):
        lines = [
            f"[{n}] {r.get('company') or '?'} - {r.get('title_en') or '?'}",
            (f"Location: {', '.join(x for x in (r.get('city'), r.get('country_code')) if x) or 'unknown'}"
             f" | work mode: {r.get('work_mode') or 'unknown'} | seniority: {r.get('seniority') or 'unknown'}"),
            (f"Fit score: {r['fit_score'] if r.get('fit_score') is not None else 'not judged'}"
             f" | status: {r.get('status') or 'no status'} | url: {r.get('url') or '-'}"),
        ]
        if r.get("summary"):
            lines.append(f"Summary: {r['summary'][:max_summary_chars]}")
        if r.get("reasons"):
            lines.append(f"Judge: {r['reasons'][:max_summary_chars]}")
        if r.get("missing_skills"):
            lines.append(f"Missing skills: {', '.join(r['missing_skills'])}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def chat_messages(question: str, rows: list[dict], history: list[dict] | None = None) -> list[dict]:
    context = build_context(rows) if rows else "(no jobs matched the search and filters)"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *(history or []),
        {"role": "user", "content": f"Jobs:\n{context}\n\nQuestion: {question}"},
    ]


def message_text(content) -> str:
    """Final answer text; reasoning models (gpt-oss) return a list of reasoning + text blocks."""
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content or [] if block.get("type") == "text")


def _placeholders(prefix: str, values: list[str], params: dict) -> str:
    names = []
    for i, value in enumerate(values):
        name = f"{prefix}{i}"
        params[name] = value
        names.append(f":{name}")
    return ", ".join(names)
