import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from job_hunter import cv, llm

NOW = datetime(2026, 10, 21, 8, 0, tzinfo=UTC)


def _job(company: str, role: str, bullets: list[str] | None = None) -> dict:
    return {"company": company, "role": role, "employment_type": None, "start": "2020", "end": "2021",
            "bullets": bullets or ["Did things"]}


DOC = {
    "name": "Jane Doe",
    "headline": "Data & AI Engineer",
    "summary": "I build data platforms.",
    "location": "Zagreb, Croatia",
    "work_authorization": None,
    "email": None,
    "phone": None,
    "links": [],
    "experience": [
        _job("A", "Data Engineer", ["• Built a RAG app", "  - Tuned Delta  ", "•"]),
        _job("B", "Analytics Engineer"),
        _job("C", "Data Engineer"),
        _job("D", "BI Analyst"),
    ],
    "skills": [{"category": "AI", "items": ["RAG", "LLMs"]}, {"category": "Platforms", "items": ["Databricks"]}],
    "certifications": [],
    "education": [],
    "languages": [],
    "trainings": [],
    "projects": [],
}


def test_bullets_are_cleaned():
    document = cv.CvDocument.model_validate(DOC)
    assert document.experience[0].bullets == ["Built a RAG app", "Tuned Delta"]


def test_roles_without_bullets():
    document = cv.CvDocument.model_validate({**DOC, "experience": [_job("A", "DE"), {**_job("B", "DA"), "bullets": []}]})
    assert document.roles_without_bullets() == ["DA at B"]


def test_match_text_uses_recent_roles_and_all_skills():
    text = cv.CvDocument.model_validate(DOC).match_text()
    assert text.startswith("Data & AI Engineer | Zagreb, Croatia")
    assert "Recent roles: Data Engineer at A; Analytics Engineer at B; Data Engineer at C" in text
    assert "BI Analyst" not in text
    assert "Skills: RAG, LLMs, Databricks" in text


@pytest.mark.parametrize(
    "broken",
    [
        {"experience": []},
        {"experience": [{**_job("A", "DE"), "start": ""}]},
        {"name": ""},
        {"unexpected": "field"},
    ],
)
def test_invalid_documents_are_rejected(broken):
    with pytest.raises(ValidationError):
        cv.CvDocument.model_validate({**DOC, **broken})


def test_file_version_is_stable_content_hash():
    assert cv.file_version(b"pdf") == cv.file_version(b"pdf")
    assert cv.file_version(b"pdf") != cv.file_version(b"pdf2")
    assert len(cv.file_version(b"pdf")) == 16


def test_to_cv_version_serializes_document():
    document = cv.CvDocument.model_validate(DOC)
    record = cv.to_cv_version("v1", "/Volumes/x/cv.pdf", "text", document, [0.1, 0.2], "llm", "emb", NOW)
    assert json.loads(record.cv_json)["name"] == "Jane Doe"
    assert record.match_text == document.match_text()
    assert record.is_current is True


def test_strict_schema_inlines_nested_models():
    """Strict mode needs every property required and no extra properties, at every level, with no $refs."""
    schema = llm.strict_schema(cv.CvDocument)
    assert "$ref" not in str(schema) and "$defs" not in str(schema)

    def check(node):
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])
            for child in node["properties"].values():
                check(child)
        if node.get("type") == "array":
            check(node["items"])

    check(schema)
    assert schema["properties"]["experience"]["items"]["properties"]["company"]["type"] == "string"


def test_elements_text_keeps_reading_order_and_drops_layout_noise():
    elements = [
        {"pos": 2, "type": "text", "content": "16 years: engineering → data & AI ", "page": 1, "confidence": 0.9},
        {"pos": 0, "type": "title", "content": "Cihan Giray ONER", "page": 1, "confidence": 0.97},
        {"pos": 3, "type": "figure", "content": "→", "page": 1, "confidence": 0.8},
        {"pos": 4, "type": "page_number", "content": "Page 1 | 3", "page": 1, "confidence": 0.99},
        {"pos": 1, "type": "section_header", "content": "DATA & AI ENGINEER", "page": 1, "confidence": 0.95},
        {"pos": 5, "type": "text", "content": "  ", "page": 2, "confidence": 0.4},
    ]
    assert cv.elements_text(elements) == "Cihan Giray ONER\nDATA & AI ENGINEER\n16 years: engineering → data & AI"
    assert [e["pos"] for e in cv.low_confidence(elements)] == [5]


def test_parse_document_sql_binds_the_path():
    sql = cv.parse_document_sql()
    assert "read_files(:path, format => 'binaryFile')" in sql
    assert "variant_explode(p:document:elements)" in sql
