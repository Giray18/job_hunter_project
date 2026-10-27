from job_hunter import chat_turns

TURN = {
    "turn_id": "t1", "asked_at": "2026-10-27T09:27:00+00:00", "question": "Remote dbt jobs?",
    "filters": {"work_modes": ["remote"]}, "index_filters_json": '{"work_mode": ["remote"]}', "history_turns": 1,
    "index": "cat.sch.jobs_index", "query_type": "HYBRID", "llm_model": "gpt-oss", "embedding_model": "gte",
    "documents": [{"page_content": "[1] A - Data Engineer", "metadata": {"doc_uri": "u", "chunk_id": "42",
                                                                         "search_score": 0.9}}],
    "answer": "[1] A fits.", "timings_ms": {"embed_ms": 100, "index_ms": 200, "sql_ms": 300, "answer_ms": 4000},
}


def test_span_plan_nests_and_times_the_steps():
    plan = {p.name: p for p in chat_turns.span_plan(TURN)}
    assert list(plan) == ["job_chat", "retrieve_jobs", "embed_query", "jobs_index.query", "job_details.sql", "answer"]
    assert (plan["job_chat"].parent, plan["job_chat"].duration_ms) == (None, 4600)
    assert (plan["retrieve_jobs"].span_type, plan["retrieve_jobs"].duration_ms) == ("RETRIEVER", 600)
    assert plan["retrieve_jobs"].outputs == TURN["documents"]
    assert (plan["jobs_index.query"].start_ms, plan["job_details.sql"].start_ms, plan["answer"].start_ms) == (100, 300, 600)
    assert plan["jobs_index.query"].outputs == [{"job_id": "42", "score": 0.9}]
    assert plan["job_chat"].outputs == {"answer": "[1] A fits."}


def test_span_plan_tolerates_missing_timings_and_documents():
    plan = chat_turns.span_plan({**TURN, "timings_ms": None, "documents": []})
    assert all(p.duration_ms == 0 for p in plan)


def test_new_turns_skip_replayed_and_sort_by_time():
    later = {**TURN, "turn_id": "t2", "asked_at": "2026-10-27T10:00:00+00:00"}
    assert [t["turn_id"] for t in chat_turns.new_turns([later, TURN], set())] == ["t1", "t2"]
    assert chat_turns.new_turns([later, TURN], {"t1"}) == [later]
    assert chat_turns.start_ns(TURN) == 1793093220 * 10**9
