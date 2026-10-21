from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from job_hunter import countries, enrichment, llm
from job_hunter.enrichment import EnrichedJob
from job_hunter.records import ApplyType, JobPosting, Seniority, WorkMode

NOW = datetime(2026, 10, 20, 8, 0, tzinfo=UTC)

VALID = {
    "language": "nl",
    "title_en": "Data Analyst",
    "description_en": "We are looking for a data analyst.",
    "summary": "Analyst role in Bruges.",
    "skills": ["SQL"],
    "seniority": "senior",
    "work_mode": "hybrid",
    "country_code": "BE",
    "city": "Bruges",
    "min_years_experience": 3,
}


def _posting(description: str = "Wij zoeken een data analist.") -> JobPosting:
    return JobPosting(
        job_id="42", url="u", title="Data analist", company="we+", location="Bruges", description=description,
        criteria={"Seniority level": "Mid"}, apply_type=ApplyType.EXTERNAL, http_status=200,
        fetched_at=NOW,
    )


def test_message_text_from_reasoning_blocks():
    content = [
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "thinking..."}]},
        {"type": "text", "text": '{"a": 1}'},
    ]
    assert llm.message_text(content) == '{"a": 1}'
    assert llm.message_text('{"a": 1}') == '{"a": 1}'


def test_parse_json_tolerates_code_fences():
    assert llm.parse_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_enriched_job_repairs_fixable_values():
    job = EnrichedJob.model_validate(
        {**VALID, "language": "NL", "skills": ["SQL", " Power BI ", "SQL", ""], "country_code": "uk", "city": " "}
    )
    assert job.language == "nl"
    assert job.skills == ["SQL", "Power BI"]
    assert job.country_code == "GB"
    assert job.city is None


def test_enriched_job_caps_skills():
    job = EnrichedJob.model_validate({**VALID, "skills": [f"s{i}" for i in range(40)]})
    assert len(job.skills) == enrichment.MAX_SKILLS


@pytest.mark.parametrize(
    "override",
    [
        {"seniority": "senior-ish"},
        {"work_mode": "fully remote"},
        {"language": "dutch"},
        {"min_years_experience": 80},
        {"country_code": "ZZ"},
        {"summary": ""},
        {"extra_field": 1},
    ],
)
def test_enriched_job_rejects_invalid_values(override):
    with pytest.raises(ValidationError):
        EnrichedJob.model_validate({**VALID, **override})


def test_to_silver_job_keeps_original_for_english_and_names_country():
    result = EnrichedJob.model_validate({**VALID, "language": "en", "description_en": "", "country_code": "HR"})
    job = enrichment.to_silver_job(_posting("Original English text."), result, "m", NOW)
    assert job.description_en == "Original English text."
    assert (job.country, job.country_code) == ("Croatia", "HR")
    assert job.seniority is Seniority.SENIOR and job.work_mode is WorkMode.HYBRID


def test_user_prompt_includes_criteria():
    prompt = enrichment.user_prompt(_posting("Build pipelines."))
    assert "- Seniority level: Mid" in prompt
    assert prompt.endswith("Description:\nBuild pipelines.")


def test_strict_schema_for_enriched_job():
    schema = llm.strict_schema(EnrichedJob)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["properties"]["seniority"]["enum"] == [s.value for s in Seniority]
    assert schema["properties"]["country_code"]["type"] == ["string", "null"]
    assert not _keys(schema) & {"$ref", "$defs", "pattern", "maxItems", "minimum", "maximum", "title"}


def _keys(node) -> set[str]:
    """All schema keywords used anywhere (property names under 'properties' are not keywords)."""
    if isinstance(node, list):
        return set().union(*(_keys(n) for n in node)) if node else set()
    if not isinstance(node, dict):
        return set()
    found = set(node)
    for key, value in node.items():
        children = value.values() if key == "properties" else [value]
        found |= set().union(*(_keys(c) for c in children)) if children else set()
    return found


def test_countries_normalization():
    assert countries.normalize_code(" hr ") == "HR"
    assert countries.name("HR") == "Croatia"
    assert countries.name(countries.normalize_code("UK")) == "United Kingdom"
    assert countries.code_from_name("Croatia") == "HR"
    assert countries.code_from_name("Republic of Croatia") == "HR"
    assert countries.code_from_name("Kosovo") == "XK"
    assert countries.code_from_name("Atlantis") is None
    with pytest.raises(ValueError):
        countries.normalize_code("ZZ")
