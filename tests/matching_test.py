"""Matching rules: the SQL filter (run on DuckDB) and its Python mirror must agree on every case."""

import re
from datetime import UTC, datetime, timedelta

import duckdb
import pytest
from pydantic import ValidationError

from job_hunter import llm, matching
from job_hunter.matching import JobFit, MatchConfig
from job_hunter.records import Priority, RemoteEligibility

NOW = datetime(2026, 10, 21, 8, 0, tzinfo=UTC)
CONFIG = MatchConfig()

# (case id, country_code, work_mode, seniority, days since first seen, expected)
CASES = [
    ("hr-onsite", "HR", "onsite", "senior", 1, True),
    ("hr-hybrid", "HR", "hybrid", "mid", 1, True),
    ("hr-remote", "HR", "remote", "lead", 1, True),
    ("hr-unknown", "HR", "unknown", "senior", 1, True),
    ("de-remote", "DE", "remote", "senior", 1, True),
    ("de-unknown", "DE", "unknown", "senior", 1, True),
    ("de-hybrid", "DE", "hybrid", "senior", 1, False),
    ("de-onsite", "DE", "onsite", "senior", 1, False),
    ("rs-remote", "RS", "remote", "senior", 1, True),
    ("xk-remote", "XK", "remote", "senior", 1, True),
    ("us-remote", "US", "remote", "senior", 1, False),
    ("tr-remote", "TR", "remote", "senior", 1, False),
    ("nocountry-remote", None, "remote", "senior", 1, True),
    ("nocountry-unknown", None, "unknown", "senior", 1, True),
    ("nocountry-onsite", None, "onsite", "senior", 1, False),
    ("junior", "HR", "onsite", "junior", 1, False),
    ("intern", "DE", "remote", "intern", 1, False),
    ("principal", "DE", "remote", "principal", 1, True),
    ("unknown-seniority", "DE", "remote", "unknown", 1, True),
    ("13-days", "HR", "onsite", "senior", 13, True),
    ("15-days", "HR", "onsite", "senior", 15, False),
    ("never-seen", "HR", "onsite", "senior", None, False),
]


@pytest.mark.parametrize(("case", "country", "work_mode", "seniority", "age_days", "expected"), CASES,
                         ids=[c[0] for c in CASES])
def test_python_filter(case, country, work_mode, seniority, age_days, expected):
    first_seen = None if age_days is None else NOW - timedelta(days=age_days)
    assert matching.passes_filter(country, work_mode, seniority, first_seen, CONFIG, NOW) is expected


def test_sql_filter_matches_python_mirror_on_duckdb():
    con = duckdb.connect()
    con.execute("CREATE TABLE e (job_id VARCHAR, country_code VARCHAR, work_mode VARCHAR, seniority VARCHAR, "
                "first_seen_at TIMESTAMPTZ)")
    for case, country, work_mode, seniority, age_days, _ in CASES:
        first_seen = None if age_days is None else NOW - timedelta(days=age_days)
        con.execute("INSERT INTO e VALUES (?, ?, ?, ?, ?)", [case, country, work_mode, seniority, first_seen])

    # Databricks uses :name parameters, DuckDB $name.
    where = re.sub(r":(\w+)", r"$\1", matching.candidate_filter_sql("e"))
    selected = {r[0] for r in con.execute(f"SELECT job_id FROM e WHERE {where}",
                                          matching.filter_params(CONFIG, NOW)).fetchall()}
    assert selected == {case for case, *_, expected in CASES if expected}


def test_shortlist_sql_uses_parameters_and_vector_function():
    sql = matching.shortlist_sql("emb", "jobs", "cv", "matches", "statuses", 20)
    assert "vector_cosine_similarity(e.embedding, cv.embedding)" in sql
    assert "LIMIT 20" in sql
    for name in ("cv_version", "target_country", "remote_countries", "allowed_seniority", "min_first_seen"):
        assert f":{name}" in sql
    # Configured values are passed as parameters, never inlined.
    assert "'HR'" not in sql and "'senior'" not in sql and "'remote'" not in sql
    assert set(re.findall(r":(\w+)", sql)) - {"cv_version"} == set(matching.filter_params(CONFIG, NOW))
    with pytest.raises(ValueError):
        matching.shortlist_sql("emb", "jobs", "cv", "matches", "statuses", 0)


def test_config_from_strings_validates():
    config = MatchConfig.from_strings("hr", "de, uk ,RS", "mid,senior", 14, 20, 60)
    assert config.target_country == "HR"
    assert config.remote_countries == ("DE", "GB", "RS")
    with pytest.raises(ValueError):
        MatchConfig.from_strings("HR", "DE,ZZ", "mid", 14, 20, 60)
    with pytest.raises(ValueError):
        MatchConfig.from_strings("HR", "DE", "mid,expert", 14, 20, 60)
    with pytest.raises(ValueError):
        MatchConfig.from_strings("HR", "DE", "mid", 14, 20, 101)


def test_match_text():
    text = matching.match_text("Data Engineer", "senior", "remote", "Croatia", ["Python", "Spark"], "Build pipelines.")
    assert text == "Data Engineer | senior | remote | Croatia\nSkills: Python, Spark\nBuild pipelines."


FIT = {"fit_score": 80, "remote_eligibility": "yes", "reasons": "Strong Databricks match.",
       "matching_skills": ["Databricks"], "missing_skills": ["Kafka"]}


def test_job_fit_caps_and_dedupes_skills():
    fit = JobFit.model_validate({**FIT, "matching_skills": ["SQL", "SQL", " "] + [f"s{i}" for i in range(30)]})
    assert fit.matching_skills[0] == "SQL" and len(fit.matching_skills) == matching.MAX_SKILL_LIST


@pytest.mark.parametrize("override", [{"fit_score": 101}, {"fit_score": -1}, {"remote_eligibility": "maybe"},
                                      {"reasons": ""}, {"extra": 1}])
def test_job_fit_rejects_invalid(override):
    with pytest.raises(ValidationError):
        JobFit.model_validate({**FIT, **override})


def test_job_fit_strict_schema():
    schema = llm.strict_schema(JobFit)
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["properties"]["remote_eligibility"]["enum"] == [e.value for e in RemoteEligibility]


@pytest.mark.parametrize(
    ("country", "work_mode", "judged", "expected"),
    [
        # Croatia with a known work mode: fixed rule, the judge's answer is ignored.
        ("HR", "remote", "no", "yes"),
        ("HR", "hybrid", "no", "yes"),
        ("HR", "onsite", "yes", "no"),
        # Croatia with unknown work mode, and all jobs abroad: the judge decides.
        ("HR", "unknown", "unclear", "unclear"),
        ("HR", "unknown", "yes", "yes"),
        ("DE", "remote", "no", "no"),
        ("DE", "unknown", "unclear", "unclear"),
        (None, "remote", "yes", "yes"),
    ],
)
def test_remote_eligibility(country, work_mode, judged, expected):
    result = matching.remote_eligibility(country, work_mode, RemoteEligibility(judged), CONFIG)
    assert result is RemoteEligibility(expected)


@pytest.mark.parametrize(
    ("country", "eligibility", "expected"),
    [
        ("HR", "yes", Priority.CROATIA),
        ("HR", "no", Priority.CROATIA),
        ("HR", "unclear", Priority.CROATIA),
        ("DE", "yes", Priority.REMOTE),
        ("DE", "unclear", Priority.REMOTE_UNCLEAR),
        ("DE", "no", Priority.EXCLUDED),
        (None, "yes", Priority.REMOTE),
        (None, "unclear", Priority.REMOTE_UNCLEAR),
    ],
)
def test_final_priority(country, eligibility, expected):
    assert matching.final_priority(country, RemoteEligibility(eligibility), CONFIG) is expected
