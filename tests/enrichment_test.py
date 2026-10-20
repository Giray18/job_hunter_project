from datetime import UTC, datetime

from job_hunter import enrichment, llm

NOW = datetime(2026, 10, 20, 8, 0, tzinfo=UTC)


def test_message_text_from_reasoning_blocks():
    content = [
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "thinking..."}]},
        {"type": "text", "text": '{"a": 1}'},
    ]
    assert llm.message_text(content) == '{"a": 1}'
    assert llm.message_text('{"a": 1}') == '{"a": 1}'


def test_parse_json_tolerates_code_fences():
    assert llm.parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert llm.parse_json('{"a": 1}') == {"a": 1}


def test_enriched_row_sanitizes_values():
    result = {
        "language": "NL",
        "title_en": "Data Analyst",
        "description_en": "We are looking for a data analyst.",
        "summary": "Analyst role in Bruges.",
        "skills": ["SQL", " Power BI ", "SQL", ""],
        "seniority": "Senior",
        "work_mode": "fully remote",
        "country": "Belgium",
        "city": "",
        "min_years_experience": 3,
    }
    assert enrichment.enriched_row("42", result, "Wij zoeken een data analist.", "m", NOW, NOW) == (
        "42", "nl", "Data Analyst", "We are looking for a data analyst.", "Analyst role in Bruges.",
        ["SQL", "Power BI"], "senior", "unknown", "Belgium", None, 3, "m", NOW, NOW,
    )


def test_enriched_row_keeps_original_for_english():
    result = {"language": "en", "description_en": "", "skills": [], "seniority": "mid", "work_mode": "remote"}
    row = enrichment.enriched_row("1", result, "Original English text.", "m", NOW, NOW)
    assert row[3] == "Original English text."


def test_user_prompt_includes_criteria():
    prompt = enrichment.user_prompt(
        {"title": "DE", "company": "Acme", "location": "Graz", "criteria": {"Seniority level": "Mid"},
         "description": "Build pipelines."}
    )
    assert "- Seniority level: Mid" in prompt
    assert prompt.endswith("Description:\nBuild pipelines.")
