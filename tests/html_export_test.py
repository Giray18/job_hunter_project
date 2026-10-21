import json
import re
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

import pytest

from job_hunter import html_export
from job_hunter.html_export import ApplicationEntry

DAY = date(2026, 10, 21)


def _entry(**overrides) -> ApplicationEntry:
    values = {
        "rank": 1, "tier": 1, "fit_score": 85, "job_id": "4476677082", "title": "Data Engineer", "company": "OIV",
        "location": "Zagreb, Croatia", "apply_url": "https://www.linkedin.com/jobs/view/4476677082/",
        "cv_file": "01_OIV_Data_Engineer_4476677082/cv.pdf",
        "cover_letter_file": "01_OIV_Data_Engineer_4476677082/cover_letter.pdf",
        "cover_letter_text": "Dear Hiring Team at OIV,\n\nI build data platforms.",
        "reasons": "Strong match.", "matching_skills": ["Databricks"], "missing_skills": ["Kafka"],
    }
    return ApplicationEntry(**{**values, **overrides})


class _Sections(HTMLParser):
    """Collects the data-* attributes of every application section."""

    def __init__(self):
        super().__init__()
        self.sections: list[dict] = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "section" and attrs.get("class") == "application":
            self.sections.append(attrs)


def _embedded(page: str) -> dict:
    match = re.search(r'<script type="application/json" id="applications-data">(.*?)</script>', page, re.DOTALL)
    return json.loads(match.group(1))


def test_sections_expose_agent_attributes():
    page = html_export.render_index(DAY, [_entry(), _entry(rank=2, job_id="2", title="BI Engineer")])
    parser = _Sections()
    parser.feed(page)
    assert [s["data-rank"] for s in parser.sections] == ["1", "2"]
    first = parser.sections[0]
    assert first["data-apply-url"] == "https://www.linkedin.com/jobs/view/4476677082/"
    assert first["data-cv-path"] == "01_OIV_Data_Engineer_4476677082/cv.pdf"
    assert first["data-cover-letter-path"] == "01_OIV_Data_Engineer_4476677082/cover_letter.pdf"


def test_embedded_json_matches_entries_and_survives_script_tags():
    hostile = _entry(title="</script><script>alert(1)</script>", cover_letter_text="Line 1\n</textarea><b>x</b>")
    page = html_export.render_index(DAY, [hostile])
    assert "<script>alert(1)</script>" not in page
    assert "</textarea><b>" not in page
    data = _embedded(page)
    assert data["date"] == "2026-10-21"
    assert data["applications"][0]["title"] == hostile.title
    assert data["applications"][0]["cover_letter_text"] == hostile.cover_letter_text


def test_local_render_uses_absolute_paths(tmp_path: Path):
    page = html_export.render_index(DAY, [_entry()], local_dir=tmp_path)
    data = _embedded(page)
    expected = str((tmp_path / "01_OIV_Data_Engineer_4476677082/cv.pdf").resolve())
    assert data["applications"][0]["cv_path"] == expected
    parser = _Sections()
    parser.feed(page)
    assert parser.sections[0]["data-cv-path"] == expected
    # Links stay relative so the page also works when the folder is moved.
    assert 'href="01_OIV_Data_Engineer_4476677082/cv.pdf"' in page


def test_json_round_trip():
    entries = [_entry(), _entry(rank=2, job_id="2", company=None, location=None)]
    assert html_export.from_json(html_export.to_json(DAY, entries)) == (DAY, entries)


def test_relative_to_day():
    day_dir = "/Volumes/c/s/cv/output/2026-10-21"
    assert html_export.relative_to_day(f"{day_dir}/01_x/cv.pdf", day_dir) == "01_x/cv.pdf"
    with pytest.raises(ValueError):
        html_export.relative_to_day("/Volumes/c/s/cv/output/2026-10-20/01_x/cv.pdf", day_dir)
