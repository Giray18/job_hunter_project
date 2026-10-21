import base64

from job_hunter import gmail_client
from job_hunter.records import JobCard


def _b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


def test_extract_job_links_canonicalizes_and_dedupes():
    text = (
        "https://www.linkedin.com/comm/jobs/view/4012345678/?trackingId=abc "
        "https://www.linkedin.com/jobs/view/4012345678 "
        "https://www.linkedin.com/comm/jobs/view/4098765432/?refId=x"
    )
    assert gmail_client.extract_job_links(text) == [
        "https://www.linkedin.com/jobs/view/4012345678/",
        "https://www.linkedin.com/jobs/view/4098765432/",
    ]


DIGEST = """data analyst - alert header: https://www.linkedin.com/comm/jobs/alerts?otpToken=secret

QC Data Analytics Specialist
PE Global
Dublin

This company is actively hiring
View job: https://www.linkedin.com/comm/jobs/view/4480995828/

---------------------------------------------------------

Business Intelligence – B.A. – DWH
Gruppo Informet
Milano
View job: https://www.linkedin.com/comm/jobs/view/4479948980/

---------------------------------------------------------

See all jobs: https://www.linkedin.com/comm/jobs/search-results/
"""


def test_extract_job_cards():
    assert gmail_client.extract_job_cards(DIGEST) == [
        JobCard(
            job_id="4480995828",
            url="https://www.linkedin.com/jobs/view/4480995828/",
            title="QC Data Analytics Specialist",
            company="PE Global",
            location="Dublin",
        ),
        JobCard(
            job_id="4479948980",
            url="https://www.linkedin.com/jobs/view/4479948980/",
            title="Business Intelligence – B.A. – DWH",
            company="Gruppo Informet",
            location="Milano",
        ),
    ]


def test_strip_url_queries_removes_tokens():
    text = "View job: https://www.linkedin.com/comm/jobs/view/1/?trackingId=a&otpToken=secret done"
    assert gmail_client.strip_url_queries(text) == "View job: https://www.linkedin.com/comm/jobs/view/1/ done"


def test_clean_snippet_removes_padding():
    assert gmail_client.clean_snippet("PE Global QC Analyst… ͏ ͏ ͏") == "PE Global QC Analyst…"


def test_extract_text_prefers_plain_part():
    payload = {
        "mimeType": "multipart/alternative",
        "parts": [
            {"mimeType": "text/plain", "body": {"data": _b64("Data Engineer at Acme")}},
            {"mimeType": "text/html", "body": {"data": _b64("<p>ignored</p>")}},
        ],
    }
    assert gmail_client.extract_text(payload) == "Data Engineer at Acme"


def test_extract_text_falls_back_to_stripped_html():
    payload = {"mimeType": "text/html", "body": {"data": _b64("<style>x{}</style><b>Data&nbsp;Engineer</b>")}}
    assert gmail_client.extract_text(payload) == "Data Engineer"
