"""Read-only Gmail access: search messages, extract text and LinkedIn job links."""

import base64
import html
import re
from email.utils import parsedate_to_datetime
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from job_hunter.records import AlertEmail, JobCard

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
LINKEDIN_JOB_ALERTS_QUERY = "from:jobalerts-noreply@linkedin.com"

# Matches linkedin.com/jobs/view/<id> and the tracking variant linkedin.com/comm/jobs/view/<id>
_JOB_LINK_RE = re.compile(r"https?://(?:[a-z]+\.)?linkedin\.com/(?:comm/)?jobs/view/(\d+)")
# Query strings on LinkedIn email links carry tracking ids and one-time login tokens (otpToken).
_URL_QUERY_RE = re.compile(r"(https?://[^\s?]+)\?\S*")
# Job cards in the alert digest are separated by lines of dashes.
_CARD_SEPARATOR_RE = re.compile(r"\n\s*-{10,}\s*\n")
# Invisible padding LinkedIn appends to the preview snippet.
_SNIPPET_PADDING_RE = re.compile(r"[\u034f\u200b-\u200d\u00ad]")


def credentials_from_refresh_token(client_id: str, client_secret: str, refresh_token: str) -> Credentials:
    """Build credentials for non-interactive use (e.g. a Databricks job reading secrets)."""
    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        client_id=client_id,
        client_secret=client_secret,
        token_uri="https://oauth2.googleapis.com/token",
        scopes=SCOPES,
    )
    creds.refresh(Request())
    return creds


def credentials_from_local_files(client_secret_path: Path, token_path: Path) -> Credentials:
    """Interactive login for local development; caches the token next to the client secret."""
    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if creds and creds.valid:
        return creds
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        # Dev-only dependency; imported lazily so the job runtime doesn't need it.
        from google_auth_oauthlib.flow import InstalledAppFlow

        flow = InstalledAppFlow.from_client_secrets_file(str(client_secret_path), SCOPES)
        creds = flow.run_local_server(port=0)
    token_path.write_text(creds.to_json())
    return creds


def build_service(creds: Credentials):
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def list_message_ids(service, query: str, max_results: int = 10) -> list[str]:
    """Return message ids matching a Gmail search query, newest first."""
    ids: list[str] = []
    page_token = None
    while len(ids) < max_results:
        resp = (
            service.users()
            .messages()
            .list(userId="me", q=query, maxResults=min(500, max_results - len(ids)), pageToken=page_token)
            .execute()
        )
        ids.extend(m["id"] for m in resp.get("messages", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return ids


def get_message(service, message_id: str) -> AlertEmail:
    """Fetch one message and flatten it into the fields we store (URL query strings stripped)."""
    msg = service.users().messages().get(userId="me", id=message_id, format="full").execute()
    headers = {h["name"].lower(): h["value"] for h in msg["payload"].get("headers", [])}
    text = strip_url_queries(extract_text(msg["payload"]))
    return AlertEmail(
        message_id=msg["id"],
        thread_id=msg["threadId"],
        received_at=parsedate_to_datetime(headers["date"]) if "date" in headers else None,
        sender=headers.get("from"),
        subject=headers.get("subject"),
        snippet=clean_snippet(msg.get("snippet", "")),
        text=text,
        job_cards=extract_job_cards(text),
    )


def extract_text(payload: dict) -> str:
    """Prefer the text/plain part; fall back to HTML with tags stripped."""
    plain = _find_part(payload, "text/plain")
    if plain:
        return plain
    return _strip_html(_html_body(payload))


def extract_job_links(text: str) -> list[str]:
    """Canonical, de-duplicated LinkedIn job URLs (tracking parameters removed)."""
    job_ids = dict.fromkeys(_JOB_LINK_RE.findall(text))
    return [_job_url(job_id) for job_id in job_ids]


def extract_job_cards(text: str) -> list[JobCard]:
    """Parse the job digest into one record per job: title, company, location and canonical URL.

    Each card in the plain-text email is the job title, company and location on separate lines,
    followed by optional badges and a "view job" line containing the link.
    """
    cards = []
    for block in _CARD_SEPARATOR_RE.split(text):
        match = _JOB_LINK_RE.search(block)
        if not match:
            continue
        before_link = block[: match.start()].splitlines()
        # Skip the line holding the link and any header lines that also contain URLs.
        lines = [line.strip() for line in before_link[:-1] if line.strip() and "http" not in line]
        title, company, location = (lines + [None, None, None])[:3]
        cards.append(
            JobCard(job_id=match.group(1), url=_job_url(match.group(1)), title=title, company=company,
                    location=location)
        )
    return cards


def strip_url_queries(text: str) -> str:
    return _URL_QUERY_RE.sub(r"\1", text)


def clean_snippet(snippet: str) -> str:
    return re.sub(r"\s+", " ", _SNIPPET_PADDING_RE.sub("", html.unescape(snippet))).strip()


def _job_url(job_id: str) -> str:
    return f"https://www.linkedin.com/jobs/view/{job_id}/"


def _html_body(payload: dict) -> str:
    return _find_part(payload, "text/html")


def _find_part(payload: dict, mime_type: str) -> str:
    if payload.get("mimeType") == mime_type and payload.get("body", {}).get("data"):
        return _decode(payload["body"]["data"])
    for part in payload.get("parts", []):
        found = _find_part(part, mime_type)
        if found:
            return found
    return ""


def _decode(data: str) -> str:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)).decode("utf-8", errors="replace")


def _strip_html(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", html.unescape(raw)).strip()
