"""Fetch public LinkedIn job postings ("About the job" / "İş ilanı hakkında") by job id.

Uses LinkedIn's public guest endpoint, which serves the same content a logged-out visitor sees.
Keep volumes low and requests spaced out: this is for personal job-alert enrichment only.
"""

import html
import re
import time
from datetime import UTC, datetime
from http import HTTPStatus

import requests

from job_hunter.records import ApplyType, JobPosting

GUEST_POSTING_URL = "https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0 Safari/537.36"
    ),
    # English UI labels keep criteria keys stable; the description itself stays in the poster's language.
    "Accept-Language": "en-US,en;q=0.9",
}

_DESCRIPTION_RE = re.compile(
    r'class="show-more-less-html__markup[^"]*">(.*?)</div>\s*(?:<button|</section>)', re.DOTALL
)
_CRITERIA_RE = re.compile(
    r'description__job-criteria-subheader">(.*?)</h3>\s*<span[^>]*>(.*?)</span>', re.DOTALL
)
_TITLE_RE = re.compile(r'<h2 class="[^"]*topcard__title[^"]*">(.*?)</h2>', re.DOTALL)
_COMPANY_RE = re.compile(r'<a class="topcard__org-name-link[^"]*"[^>]*>(.*?)</a>', re.DOTALL)
_LOCATION_RE = re.compile(r'<span class="topcard__flavor topcard__flavor--bullet">(.*?)</span>', re.DOTALL)


def fetch_job_posting(job_id: str, session: requests.Session | None = None, retries: int = 3) -> JobPosting:
    """Download and parse one job posting. Retries with backoff when LinkedIn rate-limits (429)."""
    http = session or requests.Session()
    url = GUEST_POSTING_URL.format(job_id=job_id)
    for attempt in range(retries):
        resp = http.get(url, headers=_HEADERS, timeout=30)
        if resp.status_code != HTTPStatus.TOO_MANY_REQUESTS:
            break
        time.sleep(5 * 2**attempt)

    parsed = parse_job_posting(resp.text) if resp.status_code == HTTPStatus.OK else {}
    now = datetime.now(UTC)
    return JobPosting(
        job_id=job_id,
        url=f"https://www.linkedin.com/jobs/view/{job_id}/",
        title=parsed.get("title"),
        company=parsed.get("company"),
        location=parsed.get("location"),
        description=parsed.get("description"),
        criteria=parsed.get("criteria"),
        apply_type=parsed.get("apply_type", ApplyType.UNKNOWN),
        apply_checked_at=now if parsed else None,
        http_status=resp.status_code,
        fetched_at=now,
    )


def parse_job_posting(page: str) -> dict:
    """Extract title, company, location, description text and criteria from guest posting HTML."""
    description = _DESCRIPTION_RE.search(page)
    return {
        "title": _first_text(_TITLE_RE, page),
        "company": _first_text(_COMPANY_RE, page),
        "location": _first_text(_LOCATION_RE, page),
        "description": html_to_text(description.group(1)) if description else None,
        "criteria": {_clean(k): _clean(v) for k, v in _CRITERIA_RE.findall(page)},
        "apply_type": detect_apply_type(page),
    }


def detect_apply_type(page: str) -> ApplyType:
    """The apply button's tracking name: '...apply-link-offsite' sends to the company site (external),
    '...apply-link-onsite' is LinkedIn Easy Apply. Logged-out pages label both buttons "Apply".
    A job page without any apply button no longer accepts applications."""
    offsite = "apply-link-offsite" in page or "offsite-apply" in page
    onsite = "apply-link-onsite" in page
    if offsite and not onsite:
        return ApplyType.EXTERNAL
    if onsite and not offsite:
        return ApplyType.EASY_APPLY
    if not offsite and not onsite and "top-card-layout" in page:
        return ApplyType.CLOSED
    return ApplyType.UNKNOWN


def html_to_text(fragment: str) -> str:
    """Rich-text HTML to readable plain text, keeping paragraphs and bullet points."""
    text = re.sub(r"(?i)<br\s*/?>", "\n", fragment)
    text = re.sub(r"(?i)<li[^>]*>", "\n- ", text)
    text = re.sub(r"(?i)</(p|ul|ol|li|h\d)>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    lines = [re.sub(r"[ \t ]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _first_text(pattern: re.Pattern, page: str) -> str | None:
    match = pattern.search(page)
    return _clean(match.group(1)) if match else None


def _clean(fragment: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()
