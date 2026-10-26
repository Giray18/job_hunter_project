import json
from datetime import UTC, datetime

import pytest

from job_hunter import output_eval

LETTER = "Dear Hiring Team at Expleo Group, " + "word " * 280


def test_letter_checks_pass_for_a_good_letter():
    assert output_eval.letter_checks("Expleo Group", LETTER) == {
        "letter_words": 286, "letter_length_ok": True, "letter_names_company": True, "letter_placeholders": False,
    }


@pytest.mark.parametrize("letter", ["Dear [Hiring Manager], hello", "Hi {{company}} team", "Dear <COMPANY NAME>",
                                    "Lorem ipsum dolor"])
def test_letter_checks_find_placeholders(letter):
    assert output_eval.letter_checks("Acme", letter)["letter_placeholders"]


def test_letter_checks_flag_short_letters_and_missing_company():
    checks = output_eval.letter_checks("Acme", "Dear team, I am interested.")
    assert not checks["letter_length_ok"]
    assert not checks["letter_names_company"]
    assert not output_eval.letter_checks(None, LETTER)["letter_names_company"]


def test_normalize_label():
    assert output_eval.normalize_label("Too High.", output_eval.FIT_CALIBRATION_VALUES) == "too_high"
    assert output_eval.normalize_label("'calibrated'", output_eval.FIT_CALIBRATION_VALUES) == "calibrated"
    assert output_eval.normalize_label("maybe", output_eval.FIT_CALIBRATION_VALUES) is None
    assert output_eval.normalize_label(None, output_eval.LETTER_RELEVANCE_VALUES) is None


def test_eval_record_and_quality_row_round_trip():
    row = {"job_id": "42", "cv_version": "v1", "company": "Expleo Group", "title_en": "Data Engineer",
           "description_en": "x" * 10_000, "cv_text": "cv", "fit_score": 78, "reasons": "good",
           "cover_letter_text": LETTER}
    record = output_eval.eval_record(row)
    assert len(record["inputs"]["job"]) < 6100
    assert record["outputs"] == {"fit_score": 78, "reasons": "good", "cover_letter": LETTER}

    inputs = output_eval.request_inputs(json.dumps(record["inputs"]))
    result = {"fit_score_calibration/value": "too_high", "fit_score_calibration/rationale": "asks for more",
              "cover_letter_relevance/value": "Strong", "cover_letter_relevance/rationale": "on point"}
    now = datetime(2026, 10, 26, tzinfo=UTC)
    q = output_eval.quality_row(inputs, LETTER, result, "gpt-oss", "run1", now)
    assert (q.job_id, q.cv_version, q.fit_calibration, q.letter_relevance) == ("42", "v1", "too_high", "strong")
    assert q.letter_names_company and q.letter_length_ok and not q.letter_placeholders
    assert (q.mlflow_run_id, q.evaluated_at) == ("run1", now)


def test_sample_sql_excludes_evaluated_and_validates_numbers():
    sql = output_eval.sample_sql("a_t", "m_t", "s_t", "e_t", "cv_t", "q_t", 7, 50)
    assert "NOT EXISTS (SELECT 1 FROM q_t q" in sql
    assert "INTERVAL 7 DAYS" in sql and "LIMIT 50" in sql
    with pytest.raises(ValueError):
        output_eval.sample_sql("a", "m", "s", "e", "c", "q", 0, 50)
