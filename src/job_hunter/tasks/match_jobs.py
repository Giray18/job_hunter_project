"""Job task: filter + cosine shortlist (SQL) → LLM fit judge → gold_job_matches → print today's ranking."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from job_hunter import llm, matching, ranking, storage
from job_hunter.cv import CvDocument
from job_hunter.records import JobMatch
from job_hunter.storage import Table


def main():
    parser = argparse.ArgumentParser(description="Shortlist and judge jobs against the current CV")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--model", default=llm.DEFAULT_CHAT_MODEL)
    parser.add_argument("--target-country", default="HR")
    parser.add_argument("--remote-countries", default=matching.DEFAULT_REMOTE_COUNTRIES)
    parser.add_argument("--allowed-seniority", default=matching.DEFAULT_ALLOWED_SENIORITY)
    parser.add_argument("--max-age-days", type=int, default=14)
    parser.add_argument("--max-judge", type=int, default=20)
    parser.add_argument("--croatia-min-fit", type=int, default=60)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--parallelism", type=int, default=2)
    args = parser.parse_args()

    config = matching.MatchConfig.from_strings(
        args.target_country, args.remote_countries, args.allowed_seniority, args.max_age_days, args.max_judge,
        args.croatia_min_fit,
    )

    from databricks.sdk.runtime import spark

    storage.ensure_tables(spark, args.catalog, args.schema)

    def name(table: Table) -> str:
        return storage.full_name(args.catalog, args.schema, table)

    matches_table = name(Table.GOLD_JOB_MATCHES)
    cv_row = spark.sql(f"SELECT cv_version, cv_json FROM {name(Table.SILVER_CV)} WHERE is_current").first()
    if cv_row is None:
        raise RuntimeError("No current CV in silver_cv; upload cv.pdf and run process_cv first")
    cv = CvDocument.model_validate_json(cv_row.cv_json)
    profile = matching.candidate_profile(cv)

    now = datetime.now(UTC)
    params = {**matching.filter_params(config, now), "cv_version": cv_row.cv_version}
    query = matching.shortlist_sql(name(Table.SILVER_JOB_EMBEDDINGS), name(Table.SILVER_JOBS),
                                   name(Table.SILVER_CV), matches_table, config.max_judge)
    shortlist = [r.asDict() for r in spark.sql(query, args=params).collect()]
    in_target = sum(1 for j in shortlist if j["country_code"] == config.target_country)
    print(f"Shortlisted {len(shortlist)} job(s) for CV {cv_row.cv_version} ({in_target} in {config.target_country})")

    def judge(job: dict) -> JobMatch | None:
        try:
            # Temperature 0 narrows score noise between runs (observed 10-13 points → 2-4).
            fit = llm.chat_json(matching.JUDGE_SYSTEM_PROMPT, matching.judge_prompt(profile, job), matching.JobFit,
                                model=args.model, max_tokens=3000, temperature=0)
        except RuntimeError as e:
            print(f"job {job['job_id']}: {e}")
            return None
        eligibility = matching.remote_eligibility(job["country_code"], job["work_mode"], fit.remote_eligibility, config)
        return JobMatch(
            job_id=job["job_id"],
            cv_version=cv_row.cv_version,
            similarity=float(job["similarity"]),
            priority=matching.final_priority(job["country_code"], eligibility, config),
            fit_score=fit.fit_score,
            remote_eligibility=eligibility,
            reasons=fit.reasons,
            matching_skills=fit.matching_skills,
            missing_skills=fit.missing_skills,
            model=args.model,
            judged_at=datetime.now(UTC),
        )

    with ThreadPoolExecutor(max_workers=args.parallelism) as pool:
        judged = [m for m in pool.map(judge, shortlist) if m]
    if judged:
        storage.merge(spark, storage.to_dataframe(spark, Table.GOLD_JOB_MATCHES, judged), matches_table,
                      ["job_id", "cv_version"])
    print(f"Judged {len(judged)} of {len(shortlist)} job(s) into {matches_table}")
    if shortlist and not judged:
        raise RuntimeError("No job could be judged; check the model endpoint")

    ranked = [
        ranking.from_row(r.asDict())
        for r in spark.sql(ranking.judged_jobs_sql(name(Table.SILVER_JOB_EMBEDDINGS), matches_table),
                           args={"min_first_seen": config.min_first_seen(now)}).collect()
    ]
    top = ranking.top_jobs(ranked, config.croatia_min_fit, args.top_n)
    print(f"Top {len(top)} of {len(ranked)} judged job(s) in the last {config.max_age_days} days:")
    print(ranking.describe(top, config.croatia_min_fit))


if __name__ == "__main__":
    main()
