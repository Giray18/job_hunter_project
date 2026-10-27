"""Replay of saved chat-app turns as MLflow traces (no MLflow calls here, unit-tested).

The chat app saves each turn as JSON in the chat_turns volume (its network can't reach MLflow's trace storage).
The quality job turns every new file into a trace with the turn's real timings:

    job_chat (CHAIN)                 question → answer
    ├─ retrieve_jobs (RETRIEVER)     query → the job texts the LLM got (what MLflow's retrieval judges read)
    │  ├─ embed_query (EMBEDDING)
    │  ├─ jobs_index.query (TOOL)    index, HYBRID/ANN, filters_json → hits
    │  └─ job_details.sql (TOOL)
    └─ answer (CHAT_MODEL)
"""

from dataclasses import dataclass, field
from datetime import datetime

CHAT_TRACE_NAME = "job_chat"
TURN_ID_TAG = "chat_turn_id"


@dataclass(frozen=True)
class SpanPlan:
    name: str
    span_type: str
    parent: str | None
    start_ms: float  # offset from the turn's start
    duration_ms: float
    inputs: dict
    outputs: object = None
    attributes: dict = field(default_factory=dict)


def span_plan(turn: dict) -> list[SpanPlan]:
    """Spans of one turn, parents before children, with offsets from the steps' measured durations."""
    t = turn.get("timings_ms") or {}
    embed, index, sql, answer = (float(t.get(k, 0)) for k in ("embed_ms", "index_ms", "sql_ms", "answer_ms"))
    retrieve = embed + index + sql
    docs = turn.get("documents") or []
    hits = [{"job_id": d["metadata"].get("chunk_id"), "score": d["metadata"].get("search_score")} for d in docs]
    model_attrs = {"llm_model": turn.get("llm_model"), "embedding_model": turn.get("embedding_model")}
    return [
        SpanPlan(CHAT_TRACE_NAME, "CHAIN", None, 0, retrieve + answer,
                 {"question": turn["question"], "filters": turn.get("filters")}, {"answer": turn["answer"]},
                 {k: v for k, v in model_attrs.items() if v}),
        SpanPlan("retrieve_jobs", "RETRIEVER", CHAT_TRACE_NAME, 0, retrieve, {"query": turn["question"]}, docs),
        SpanPlan("embed_query", "EMBEDDING", "retrieve_jobs", 0, embed,
                 {"model": turn.get("embedding_model"), "input": turn["question"]}),
        SpanPlan("jobs_index.query", "TOOL", "retrieve_jobs", embed, index,
                 {"index": turn.get("index"), "query_type": turn.get("query_type"),
                  "filters_json": turn.get("index_filters_json")}, hits),
        SpanPlan("job_details.sql", "TOOL", "retrieve_jobs", embed + index, sql, {"job_ids": [h["job_id"] for h in hits]},
                 {"rows": len(docs)}),
        SpanPlan("answer", "CHAT_MODEL", CHAT_TRACE_NAME, retrieve, answer,
                 {"model": turn.get("llm_model"), "question": turn["question"],
                  "history_turns": turn.get("history_turns", 0)}, turn["answer"]),
    ]


def start_ns(turn: dict) -> int:
    return int(datetime.fromisoformat(turn["asked_at"]).timestamp() * 1e9)


def new_turns(turns: list[dict], replayed_ids: set[str]) -> list[dict]:
    """Turns without a trace yet, oldest first."""
    return sorted((t for t in turns if t["turn_id"] not in replayed_ids), key=lambda t: t["asked_at"])
