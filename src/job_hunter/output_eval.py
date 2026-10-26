"""Quality checks of the pipeline's LLM outputs, run weekly with MLflow (no Databricks or MLflow calls here).

Two LLM judges (MLflow `make_judge`) review the fit judge's score and the cover letter's relevance; code checks
cover the letter's format. Letters are deliberately not checked for factual claims: extrapolation is accepted.
"""

import json
import re
from datetime import datetime

from job_hunter.records import OutputQuality
from job_hunter.tailoring import LETTER_MAX_WORDS, LETTER_MIN_WORDS, company_mentioned, word_count

FIT_CALIBRATION = "fit_score_calibration"
LETTER_RELEVANCE = "cover_letter_relevance"
FIT_CALIBRATION_VALUES = ("calibrated", "too_high", "too_low")
LETTER_RELEVANCE_VALUES = ("strong", "adequate", "weak")
MAX_JOB_CHARS = 6000
MAX_CV_CHARS = 8000

FIT_JUDGE_INSTRUCTIONS = """You review another model's job-fit assessment for a personal job search.
Job posting and candidate CV: {{ inputs }}
The other model's assessment (fit_score 0-100, reasons): {{ outputs }}
Its scale: 90-100 near-perfect, 70-89 strong with minor gaps, 50-69 partial (some must-haves missing), 0-49 weak.
Is the fit_score consistent with the job's must-have requirements and the CV's evidence?
Answer exactly one of: 'calibrated', 'too_high', 'too_low'."""

LETTER_JUDGE_INSTRUCTIONS = """You review a cover letter written for one job.
Job posting and candidate CV: {{ inputs }}
Cover letter: {{ outputs }}
Does the letter address this job's main requirements and connect them to the candidate's experience?
Judge relevance and focus only; do not penalise claims that go beyond the CV.
Answer exactly one of: 'strong', 'adequate', 'weak'."""

# "[Name]", "{{company}}", "<COMPANY>", "lorem ipsum": text a generator left unfilled.
_PLACEHOLDER = re.compile(r"\[[A-Za-z][^\]\n]{0,40}\]|\{\{[^}]*\}\}|<[A-Z][A-Z _]{2,30}>|lorem ipsum", re.IGNORECASE)


def letter_checks(company: str | None, letter: str) -> dict:
    """Format checks of a cover letter, without an LLM."""
    words = word_count(letter)
    return {
        "letter_words": words,
        "letter_length_ok": LETTER_MIN_WORDS <= words <= LETTER_MAX_WORDS,
        "letter_names_company": bool(company) and company_mentioned(company, letter),
        "letter_placeholders": bool(_PLACEHOLDER.search(letter)),
    }


def sample_sql(applications: str, matches: str, jobs: str, embeddings: str, cvs: str, quality: str,
               lookback_days: int, limit: int) -> str:
    """Applications of the last `lookback_days` not evaluated yet, with the job, the judged fit and the CV text."""
    if not isinstance(lookback_days, int) or lookback_days <= 0 or not isinstance(limit, int) or limit <= 0:
        raise ValueError("lookback_days and limit must be positive ints")
    return f"""
        SELECT a.job_id, a.cv_version, e.company, e.title_en, s.description_en, a.cover_letter_text,
               m.fit_score, m.reasons, cv.cv_text
        FROM {applications} a
        JOIN {jobs} s ON s.job_id = a.job_id
        JOIN {embeddings} e ON e.job_id = a.job_id
        JOIN {matches} m ON m.job_id = a.job_id AND m.cv_version = a.cv_version
        JOIN {cvs} cv ON cv.cv_version = a.cv_version
        WHERE a.created_at >= current_timestamp() - INTERVAL {lookback_days} DAYS
          AND a.cover_letter_text IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM {quality} q WHERE q.job_id = a.job_id AND q.cv_version = a.cv_version)
        QUALIFY row_number() OVER (PARTITION BY a.job_id, a.cv_version ORDER BY m.judged_at DESC) = 1
        ORDER BY a.created_at DESC
        LIMIT {limit}
    """


def eval_record(row: dict) -> dict:
    """One mlflow.genai.evaluate record: what the judges see (inputs) and the pipeline's outputs."""
    return {
        "inputs": {
            "job_id": row["job_id"],
            "cv_version": row["cv_version"],
            "company": row.get("company"),
            "job": f"{row.get('title_en')} at {row.get('company')}\n\n{(row.get('description_en') or '')[:MAX_JOB_CHARS]}",
            "cv": (row.get("cv_text") or "")[:MAX_CV_CHARS],
        },
        "outputs": {
            "fit_score": row.get("fit_score"),
            "reasons": row.get("reasons"),
            "cover_letter": row.get("cover_letter_text") or "",
        },
    }


def normalize_label(value, allowed: tuple[str, ...]) -> str | None:
    """A judge's answer as one of the allowed labels ('Too High.' → 'too_high'), else None."""
    if value is None:
        return None
    label = re.sub(r"[^a-z]+", "_", str(value).lower()).strip("_")
    return label if label in allowed else None


def request_inputs(request) -> dict:
    """The `request` column of an evaluation result: the record's inputs, as a dict or JSON text."""
    if isinstance(request, str):
        request = json.loads(request)
    return request.get("inputs", request) if isinstance(request, dict) else {}


def quality_row(inputs: dict, cover_letter: str, result: dict, judge_model: str, run_id: str,
                evaluated_at: datetime) -> OutputQuality:
    """gold_output_quality row from one evaluation result (`<scorer>/value` and `<scorer>/rationale` keys)."""
    checks = letter_checks(inputs.get("company"), cover_letter)
    return OutputQuality(
        job_id=inputs["job_id"],
        cv_version=inputs["cv_version"],
        fit_calibration=normalize_label(result.get(f"{FIT_CALIBRATION}/value"), FIT_CALIBRATION_VALUES),
        fit_calibration_rationale=result.get(f"{FIT_CALIBRATION}/rationale"),
        letter_relevance=normalize_label(result.get(f"{LETTER_RELEVANCE}/value"), LETTER_RELEVANCE_VALUES),
        letter_relevance_rationale=result.get(f"{LETTER_RELEVANCE}/rationale"),
        judge_model=judge_model,
        mlflow_run_id=run_id,
        evaluated_at=evaluated_at,
        **checks,
    )
