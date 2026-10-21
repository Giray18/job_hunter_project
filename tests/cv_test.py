import json
from datetime import UTC, datetime

from job_hunter import cv

NOW = datetime(2026, 10, 21, 8, 0, tzinfo=UTC)

CV = {
    "name": "Jane Doe",
    "headline": "Data & AI Engineer",
    "summary": "I build data platforms.",
    "location": "Zagreb, Croatia",
    "experience": [
        {"company": "A", "role": "Data Engineer", "start": "2024", "end": "Ongoing", "bullets": []},
        {"company": "B", "role": "Analytics Engineer", "start": "2023", "end": "2024", "bullets": []},
        {"company": "C", "role": "Data Engineer", "start": "2022", "end": "2023", "bullets": []},
        {"company": "D", "role": "BI Analyst", "start": "2019", "end": "2022", "bullets": []},
    ],
    "skills": [{"category": "AI", "items": ["RAG", "LLMs"]}, {"category": "Platforms", "items": ["Databricks"]}],
}


def test_match_text_uses_recent_roles_and_all_skills():
    text = cv.match_text(CV)
    assert text.startswith("Data & AI Engineer | Zagreb, Croatia")
    assert "Recent roles: Data Engineer at A; Analytics Engineer at B; Data Engineer at C" in text
    assert "BI Analyst" not in text
    assert "Skills: RAG, LLMs, Databricks" in text


def test_file_version_is_stable_content_hash():
    assert cv.file_version(b"pdf") == cv.file_version(b"pdf")
    assert cv.file_version(b"pdf") != cv.file_version(b"pdf2")
    assert len(cv.file_version(b"pdf")) == 16


def test_cv_row_serializes_json_and_marks_current():
    row = cv.cv_row("v1", "/Volumes/x/cv.pdf", "text", CV, [0.1, 0.2], "llm", "emb", NOW)
    assert row[0] == "v1"
    assert json.loads(row[3])["name"] == "Jane Doe"
    assert row[4] == cv.match_text(CV)
    assert row[5] == [0.1, 0.2]
    assert row[-1] is True


def test_normalize_strips_bullet_symbols_and_flags_empty_roles():
    data = {
        "experience": [
            {"role": "DE", "company": "A", "bullets": ["• Built a RAG app", "  - Tuned Delta  ", "•"]},
            {"role": "DA", "company": "B", "bullets": []},
        ]
    }
    cv.normalize(data)
    assert data["experience"][0]["bullets"] == ["Built a RAG app", "Tuned Delta"]
    assert cv.roles_without_bullets(data) == ["DA at B"]


def test_schema_is_strict():
    """Strict JSON-schema mode needs every property required and no extra properties, at every level."""

    def check(node):
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])
            for child in node["properties"].values():
                check(child)
        if node.get("type") == "array":
            check(node["items"])

    check(cv.SCHEMA)
