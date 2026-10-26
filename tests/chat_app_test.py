import json
import re
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "job_hunter_app"))

import chat_logic
from chat_logic import NO_STATUS, Filters


def test_index_filters_only_index_columns():
    filters = Filters(country_codes=["HR", "RS"], work_modes=["remote"], statuses=["applied"], min_fit=70)
    assert json.loads(chat_logic.index_filters_json(filters)) == {"country_code": ["HR", "RS"], "work_mode": ["remote"]}
    assert chat_logic.index_filters_json(Filters()) is None


def test_jobs_sql_binds_every_value_as_parameter():
    sql, params = chat_logic.jobs_sql("cat", "sch", Filters(country_codes=["HR"], statuses=["applied", NO_STATUS],
                                                            min_fit=60), job_ids=["1", "2"])
    assert params == {"min_fit": 60, "max_age_days": 30, "cc0": "HR", "st0": "applied", "id0": "1", "id1": "2"}
    assert "e.country_code IN (:cc0)" in sql
    assert "(st.status IN (:st0) OR st.status IS NULL)" in sql
    assert "e.job_id IN (:id0, :id1)" in sql
    assert "'HR'" not in sql


def test_jobs_sql_with_no_search_hits_returns_nothing():
    sql, _ = chat_logic.jobs_sql("cat", "sch", Filters(), job_ids=[])
    assert "AND FALSE" in sql


def test_jobs_sql_runs_and_filters(tmp_path):
    con = duckdb.connect()
    con.execute("ATTACH ':memory:' AS cat")
    con.execute("CREATE SCHEMA cat.sch")
    con.execute("""CREATE TABLE cat.sch.silver_job_embeddings (job_id VARCHAR, title_en VARCHAR, company VARCHAR,
                   url VARCHAR, country_code VARCHAR, work_mode VARCHAR, seniority VARCHAR, first_seen_at TIMESTAMP)""")
    con.execute("CREATE TABLE cat.sch.silver_jobs (job_id VARCHAR, city VARCHAR, summary VARCHAR)")
    con.execute("""CREATE TABLE cat.sch.gold_job_matches (job_id VARCHAR, fit_score INT, priority VARCHAR,
                   reasons VARCHAR, missing_skills VARCHAR[], judged_at TIMESTAMP)""")
    con.execute("""CREATE TABLE cat.sch.gold_applications (job_id VARCHAR, cv_pdf_path VARCHAR,
                   cover_letter_pdf_path VARCHAR, created_at TIMESTAMP)""")
    con.execute("CREATE TABLE cat.sch.gold_application_status (job_id VARCHAR, status VARCHAR)")
    con.execute("""INSERT INTO cat.sch.silver_job_embeddings VALUES
        ('1', 'Data Engineer', 'A', 'u1', 'HR', 'remote', 'senior', now()),
        ('2', 'Analyst', 'B', 'u2', 'RS', 'hybrid', 'mid', now()),
        ('3', 'Old job', 'C', 'u3', 'HR', 'remote', 'senior', now() - INTERVAL 60 DAY)""")
    con.execute("INSERT INTO cat.sch.silver_jobs VALUES ('1', 'Zagreb', 's1'), ('2', 'Belgrade', 's2')")
    con.execute("""INSERT INTO cat.sch.gold_job_matches VALUES
        ('1', 50, 'croatia', 'old judgement', [], now() - INTERVAL 2 DAY),
        ('1', 80, 'croatia', 'good', ['dbt'], now()),
        ('2', 65, 'remote', 'ok', [], now())""")
    con.execute("INSERT INTO cat.sch.gold_applications VALUES ('1', '/Volumes/x/cv.pdf', '/Volumes/x/cl.pdf', now())")
    con.execute("INSERT INTO cat.sch.gold_application_status VALUES ('2', 'applied')")

    def run(filters, job_ids=None):
        sql, params = chat_logic.jobs_sql("cat", "sch", filters, job_ids)
        # Dialect differences only: Databricks interval function, :name parameters and backtick quoting.
        sql = sql.replace("make_interval(0, 0, 0, :max_age_days)", "to_days(CAST(:max_age_days AS INTEGER))")
        sql = re.sub(r":(\w+)", r"$\1", sql.replace("`", '"'))
        cur = con.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]

    rows = run(Filters())
    assert [r["job_id"] for r in rows] == ["1", "2"]
    assert rows[0]["fit_score"] == 80 and rows[0]["cv_pdf_path"] == "/Volumes/x/cv.pdf"
    assert [r["job_id"] for r in run(Filters(statuses=[NO_STATUS]))] == ["1"]
    assert [r["job_id"] for r in run(Filters(min_fit=70))] == ["1"]
    assert [r["job_id"] for r in run(Filters(work_modes=["hybrid"]))] == ["2"]
    assert [r["job_id"] for r in run(Filters(), job_ids=["2"])] == ["2"]
    assert [r["job_id"] for r in run(Filters(max_age_days=90))] == ["1", "2", "3"]


def test_order_like_keeps_search_order():
    rows = [{"job_id": "a"}, {"job_id": "b"}, {"job_id": "c"}]
    assert [r["job_id"] for r in chat_logic.order_like(rows, ["c", "a", "b"])] == ["c", "a", "b"]


def test_chat_messages_number_jobs_and_keep_history():
    rows = [{"job_id": "1", "company": "Codat", "title_en": "Senior Data Engineer", "country_code": "GB",
             "work_mode": "remote", "fit_score": 82, "status": "applied", "missing_skills": ["Kafka"]}]
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    messages = chat_logic.chat_messages("Which remote jobs?", rows, history)
    assert messages[0]["role"] == "system" and messages[1:3] == history
    user = messages[-1]["content"]
    assert "[1] Codat - Senior Data Engineer" in user
    assert "Fit score: 82 | status: applied" in user and "Missing skills: Kafka" in user
    assert user.endswith("Question: Which remote jobs?")


def test_message_text_skips_reasoning_blocks():
    content = [{"type": "reasoning", "summary": [{"text": "thinking"}]}, {"type": "text", "text": "Answer"}]
    assert chat_logic.message_text(content) == "Answer"
    assert chat_logic.message_text("plain") == "plain"


def test_parse_hits_keeps_rank_order_and_score():
    hits = chat_logic.parse_hits(["job_id", "title_en", "company", "score"],
                                 [["2", "Analyst", "B", 0.031], ["1", "Data Engineer", "A", 0.029]])
    assert [h["job_id"] for h in hits] == ["2", "1"]
    assert hits[0]["score"] == 0.031
    assert chat_logic.parse_hits(["job_id"], None) == []


def test_with_hits_orders_by_hit_and_drops_filtered_jobs():
    hits = [{"job_id": "3", "score": 0.9}, {"job_id": "1", "score": 0.8}, {"job_id": "2", "score": 0.7}]
    rows = [{"job_id": "1", "fit_score": 90}, {"job_id": "3", "fit_score": 40}]  # job 2 removed by the SQL filters
    assert chat_logic.with_hits(rows, hits) == [
        {"job_id": "3", "fit_score": 40, "search_rank": 1, "search_score": 0.9},
        {"job_id": "1", "fit_score": 90, "search_rank": 2, "search_score": 0.8},
    ]
