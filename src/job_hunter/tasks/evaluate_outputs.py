"""Job task: recent applications → MLflow evaluation (2 LLM judges + code checks) → MLflow run + gold_output_quality."""

import argparse
import os
from datetime import UTC, datetime

from job_hunter import llm, output_eval, storage
from job_hunter.records import OutputQuality
from job_hunter.storage import Table


def main():
    parser = argparse.ArgumentParser(description="Judge recent fit scores and cover letters with MLflow")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--experiment-id", required=True, help="MLflow experiment for the evaluation runs")
    parser.add_argument("--model", default=llm.DEFAULT_CHAT_MODEL, help="Judge model (Foundation Model endpoint)")
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    from databricks.sdk.runtime import spark

    storage.ensure_tables(spark, args.catalog, args.schema, only=[Table.GOLD_OUTPUT_QUALITY])

    def name(table: Table) -> str:
        return storage.full_name(args.catalog, args.schema, table)

    rows = [r.asDict() for r in spark.sql(output_eval.sample_sql(
        name(Table.GOLD_APPLICATIONS), name(Table.GOLD_JOB_MATCHES), name(Table.SILVER_JOBS),
        name(Table.SILVER_JOB_EMBEDDINGS), name(Table.SILVER_CV), name(Table.GOLD_OUTPUT_QUALITY),
        args.lookback_days, args.limit)).collect()]
    if not rows:
        print(f"No unevaluated applications in the last {args.lookback_days} days")
        return
    print(f"Evaluating {len(rows)} application(s) with judge model {args.model}")

    # Few parallel judge calls: Free Edition Foundation Model rate limits.
    os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_WORKERS", "2")
    import mlflow
    from mlflow.entities import Feedback
    from mlflow.genai.judges import make_judge
    from mlflow.genai.scorers import scorer

    judge_model = f"databricks:/{args.model}"
    fit_judge = make_judge(name=output_eval.FIT_CALIBRATION, instructions=output_eval.FIT_JUDGE_INSTRUCTIONS,
                           model=judge_model, feedback_value_type=str)
    letter_judge = make_judge(name=output_eval.LETTER_RELEVANCE, instructions=output_eval.LETTER_JUDGE_INSTRUCTIONS,
                              model=judge_model, feedback_value_type=str)

    @scorer
    def letter_words(inputs, outputs) -> int:
        return output_eval.letter_checks(inputs["company"], outputs["cover_letter"])["letter_words"]

    @scorer
    def letter_format(inputs, outputs) -> Feedback:
        checks = output_eval.letter_checks(inputs["company"], outputs["cover_letter"])
        ok = checks["letter_length_ok"] and checks["letter_names_company"] and not checks["letter_placeholders"]
        return Feedback(value=ok, rationale=", ".join(f"{k}={v}" for k, v in checks.items()))

    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(experiment_id=args.experiment_id)
    with mlflow.start_run(run_name=f"outputs {datetime.now(UTC):%Y-%m-%d}") as run:
        mlflow.log_params({"judge_model": args.model, "lookback_days": args.lookback_days, "applications": len(rows)})
        result = mlflow.genai.evaluate(data=[output_eval.eval_record(r) for r in rows],
                                       scorers=[fit_judge, letter_judge, letter_words, letter_format])

    letters = {(r["job_id"], r["cv_version"]): r["cover_letter_text"] for r in rows}
    now = datetime.now(UTC)
    quality: list[OutputQuality] = []
    for record in result.result_df.to_dict("records"):
        inputs = output_eval.request_inputs(record["request"])
        quality.append(output_eval.quality_row(inputs, letters[(inputs["job_id"], inputs["cv_version"])], record,
                                               args.model, run.info.run_id, now))
    storage.merge(spark, storage.to_dataframe(spark, Table.GOLD_OUTPUT_QUALITY, quality),
                  name(Table.GOLD_OUTPUT_QUALITY), ["job_id", "cv_version"])
    print(f"MLflow run {run.info.run_id}: {result.metrics}")
    for q in quality:
        print(f"  {q.job_id}: fit {q.fit_calibration}, letter {q.letter_relevance}, {q.letter_words} words, "
              f"company named {q.letter_names_company}, placeholders {q.letter_placeholders}")


if __name__ == "__main__":
    main()
