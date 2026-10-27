"""Job task: chat turns saved by the app (chat_turns volume) → MLflow traces → MLflow retrieval judges.

Each new turn is replayed as a trace with its real timings (see job_hunter.chat_turns); only those new traces are
scored, with RetrievalGroundedness, RelevanceToQuery and a citation Guidelines judge.
"""

import argparse
import json
import os

from job_hunter import chat_turns, llm

CITATION_GUIDELINE = (
    "The response refers to the jobs it discusses by their [n] number from the retrieved list, or says that "
    "none of the retrieved jobs answer the question."
)


def list_turn_files(workspace, directory: str) -> list[str]:
    """Paths of all turn files (<dir>/<date>/<turn_id>.json)."""
    paths = []
    for entry in workspace.files.list_directory_contents(directory):
        if entry.is_directory:
            paths += list_turn_files(workspace, entry.path)
        elif entry.path.endswith(".json"):
            paths.append(entry.path)
    return paths


def replay(mlflow, turn: dict, experiment_id: str) -> str:
    """Create the turn's trace with its original timestamps; returns the trace id."""
    t0 = chat_turns.start_ns(turn)
    spans = {}
    for plan in chat_turns.span_plan(turn):
        parent = spans[plan.parent][0] if plan.parent else None
        root_only = {} if parent else {"experiment_id": experiment_id, "tags": {chat_turns.TURN_ID_TAG: turn["turn_id"]}}
        span = mlflow.start_span_no_context(name=plan.name, span_type=plan.span_type, parent_span=parent,
                                            inputs=plan.inputs, attributes=plan.attributes or None,
                                            start_time_ns=t0 + int(plan.start_ms * 1e6), **root_only)
        spans[plan.name] = (span, plan)
    for span, plan in reversed(list(spans.values())):  # children end before their parents
        span.end(outputs=plan.outputs, end_time_ns=t0 + int((plan.start_ms + plan.duration_ms) * 1e6))
    return spans[chat_turns.CHAT_TRACE_NAME][0].trace_id


def main():
    parser = argparse.ArgumentParser(description="Replay saved chat turns as MLflow traces and score them")
    parser.add_argument("--turns-dir", required=True, help="e.g. /Volumes/<catalog>/<schema>/chat_turns")
    parser.add_argument("--experiment-id", required=True, help="MLflow experiment for the chat traces")
    parser.add_argument("--model", default=llm.DEFAULT_CHAT_MODEL, help="Judge model (Foundation Model endpoint)")
    parser.add_argument("--limit", type=int, default=100, help="Most turns replayed and scored per run")
    args = parser.parse_args()

    # Few parallel judge calls: Free Edition Foundation Model rate limits.
    os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_WORKERS", "2")
    import mlflow
    from databricks.sdk import WorkspaceClient
    from mlflow.genai.scorers import Guidelines, RelevanceToQuery, RetrievalGroundedness

    workspace = WorkspaceClient()
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(experiment_id=args.experiment_id)

    existing = mlflow.search_traces(experiment_ids=[args.experiment_id], max_results=5000,
                                    filter_string=f"tags.`mlflow.traceName` = '{chat_turns.CHAT_TRACE_NAME}'")
    replayed = {(tags or {}).get(chat_turns.TURN_ID_TAG) for tags in existing.get("tags", [])}
    files = [p for p in list_turn_files(workspace, args.turns_dir)
             if os.path.basename(p).removesuffix(".json") not in replayed]
    turns = [json.loads(workspace.files.download(p).contents.read()) for p in files]
    turns = chat_turns.new_turns(turns, replayed)[: args.limit]
    if not turns:
        print(f"No new chat turns in {args.turns_dir} ({len(replayed - {None})} already traced)")
        return

    trace_ids = [replay(mlflow, turn, args.experiment_id) for turn in turns]
    mlflow.flush_trace_async_logging()
    print(f"Replayed {len(trace_ids)} chat turn(s) as traces in experiment {args.experiment_id}")

    traces = mlflow.search_traces(experiment_ids=[args.experiment_id], max_results=5000,
                                  filter_string=f"tags.`mlflow.traceName` = '{chat_turns.CHAT_TRACE_NAME}'")
    traces = traces[traces["trace_id"].isin(trace_ids)]
    judge_model = f"databricks:/{args.model}"
    with mlflow.start_run(run_name=f"chat {turns[-1]['asked_at'][:10]}"):
        mlflow.log_params({"judge_model": args.model, "turns": len(trace_ids)})
        result = mlflow.genai.evaluate(data=traces, scorers=[
            RetrievalGroundedness(model=judge_model),
            RelevanceToQuery(model=judge_model),
            Guidelines(name="cites_jobs", guidelines=CITATION_GUIDELINE, model=judge_model),
        ])
    print(f"MLflow run {result.run_id}: {result.metrics}")


if __name__ == "__main__":
    main()
