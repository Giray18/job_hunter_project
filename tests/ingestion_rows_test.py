from datetime import UTC, datetime

from job_hunter.tasks.fetch_postings import posting_row
from job_hunter.tasks.ingest_emails import email_row

NOW = datetime(2026, 10, 20, 8, 0, tzinfo=UTC)


def test_email_row_matches_schema_order():
    msg = {
        "id": "m1",
        "thread_id": "t1",
        "date": NOW,
        "from": "jobalerts-noreply@linkedin.com",
        "subject": "Data Engineer",
        "snippet": "snippet",
        "text": "text",
        "job_links": ["ignored"],
        "job_cards": [
            {"job_id": "1", "url": "u", "title": "DE", "company": "Acme", "location": "Graz"},
        ],
    }
    assert email_row(msg, NOW) == (
        "m1",
        "t1",
        NOW,
        "jobalerts-noreply@linkedin.com",
        "Data Engineer",
        "snippet",
        "text",
        [("1", "u", "DE", "Acme", "Graz")],
        NOW,
    )


def test_posting_row_for_failed_fetch_has_nulls():
    posting = {"job_id": "1", "url": "u", "http_status": 404, "fetched_at": NOW}
    assert posting_row(posting) == ("1", "u", None, None, None, None, None, 404, NOW)
