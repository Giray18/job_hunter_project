"""Job task: the chat app's recent MLflow traces → MLflow retrieval judges (groundedness, relevance, citations)."""

import argparse
import os
import time
from datetime import UTC, datetime

from job_hunter import llm

CHAT_TRACE_NAME = "job_chat"
CITATION_GUIDELINE = (
    "The response refers to the jobs it discusses by their [n] number from the retrieved list, or says that "
    "none of the retrieved jobs answer the question."
)


def main():
    parser = argparse.ArgumentParser(description="Score recent chat app answers with MLflow judges")
    parser.add_argument("--experiment-id", required=True, help="MLflow experiment the chat app traces to")
    parser.add_argument("--model", default=llm.DEFAULT_CHAT_MODEL, help="Judge model (Foundation Model endpoint)")
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    # Few parallel judge calls: Free Edition Foundation Model rate limits.
    os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_WORKERS", "2")
    import mlflow
    from mlflow.genai.scorers import Guidelines, RelevanceToQuery, RetrievalGroundedness

    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(experiment_id=args.experiment_id)
    since_ms = int((time.time() - args.lookback_days * 86400) * 1000)
    traces = mlflow.search_traces(
        experiment_ids=[args.experiment_id],
        filter_string=f"attributes.timestamp_ms > {since_ms} AND tags.`mlflow.traceName` = '{CHAT_TRACE_NAME}'",
        max_results=args.limit,
    )
    if traces.empty:
        print(f"No chat traces in the last {args.lookback_days} days")
        return
    print(f"Scoring {len(traces)} chat trace(s) with judge model {args.model}")

    judge_model = f"databricks:/{args.model}"
    with mlflow.start_run(run_name=f"chat {datetime.now(UTC):%Y-%m-%d}"):
        mlflow.log_params({"judge_model": args.model, "lookback_days": args.lookback_days, "traces": len(traces)})
        result = mlflow.genai.evaluate(data=traces, scorers=[
            RetrievalGroundedness(model=judge_model),
            RelevanceToQuery(model=judge_model),
            Guidelines(name="cites_jobs", guidelines=CITATION_GUIDELINE, model=judge_model),
        ])
    print(f"MLflow run {result.run_id}: {result.metrics}")


if __name__ == "__main__":
    main()
