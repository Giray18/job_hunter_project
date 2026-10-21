"""Job task: top-ranked jobs → tailored CV + cover letter (LLM, fact-checked) → PDFs in the UC volume
→ gold_applications."""

import argparse
import os
from datetime import UTC, datetime

from job_hunter import html_export, llm, matching, pdf_render, ranking, storage, tailoring
from job_hunter.cv import CvDocument
from job_hunter.records import Application
from job_hunter.storage import Table


def main():
    parser = argparse.ArgumentParser(description="Generate a tailored CV and cover letter for each top job")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--output-dir", required=True, help="e.g. /Volumes/<catalog>/<schema>/cv/output")
    parser.add_argument("--model", default=llm.DEFAULT_CHAT_MODEL)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--priority-min-fit", type=int, default=60)
    parser.add_argument("--max-age-days", type=int, default=14)
    parser.add_argument("--availability-note", default=tailoring.DEFAULT_AVAILABILITY_NOTE)
    parser.add_argument("--b2b-note", default=tailoring.DEFAULT_B2B_NOTE)
    args = parser.parse_args()

    from databricks.sdk.runtime import spark

    storage.ensure_tables(spark, args.catalog, args.schema)

    def name(table: Table) -> str:
        return storage.full_name(args.catalog, args.schema, table)

    applications_table = name(Table.GOLD_APPLICATIONS)
    cv_row = spark.sql(f"SELECT cv_version, cv_json FROM {name(Table.SILVER_CV)} WHERE is_current").first()
    if cv_row is None:
        raise RuntimeError("No current CV in silver_cv; upload cv.pdf and run process_cv first")
    source = CvDocument.model_validate_json(cv_row.cv_json)

    now = datetime.now(UTC)
    config = matching.MatchConfig(max_age_days=args.max_age_days, priority_min_fit=args.priority_min_fit)
    ranked = [
        ranking.from_row(r.asDict())
        for r in spark.sql(ranking.judged_jobs_sql(name(Table.SILVER_JOB_EMBEDDINGS), name(Table.GOLD_JOB_MATCHES)),
                           args={"min_first_seen": config.min_first_seen(now)}).collect()
    ]
    top = ranking.top_jobs(ranked, args.priority_min_fit, args.top_n)
    done = {
        r.job_id
        for r in spark.sql(f"SELECT job_id FROM {applications_table} WHERE cv_version = :cv_version",
                           args={"cv_version": cv_row.cv_version}).collect()
    }
    todo = [(position, job) for position, job in enumerate(top, 1) if job.job_id not in done]
    print(f"Top {len(top)} job(s); {len(todo)} need a tailored CV and cover letter")
    if not todo:
        _write_day_page(spark, name, args.output_dir, datetime.now(UTC).date())
        return

    details = _job_details(spark, name, [job.job_id for _, job in todo], cv_row.cv_version)
    closing = tailoring.closing_paragraph(args.availability_note, args.b2b_note)

    created = 0
    for position, ranked_job in todo:
        job = details[ranked_job.job_id]
        try:
            application = _create_application(source, cv_row.cv_version, position, ranked_job, job, closing, args)
        except Exception as e:  # noqa: BLE001 - one failed job must not stop the others
            print(f"#{position} {ranked_job.title_en} | {ranked_job.company}: FAILED {e}")
            continue
        # Merge per job so finished applications are kept even if a later one fails.
        storage.merge(spark, storage.to_dataframe(spark, Table.GOLD_APPLICATIONS, [application]),
                      applications_table, ["job_id", "cv_version"])
        created += 1
        print(f"#{position} {ranked_job.title_en} | {ranked_job.company} -> {os.path.dirname(application.cv_pdf_path)}")

    print(f"Created {created} of {len(todo)} application(s) in {applications_table}")
    if created:
        _write_day_page(spark, name, args.output_dir, datetime.now(UTC).date())
    else:
        raise RuntimeError("No application could be created; see the errors above")


def _write_day_page(spark, name, output_dir: str, day) -> None:
    """index.html + applications.json for every application in today's output folder."""
    day_dir = pdf_render.day_dir(output_dir, day)
    rows = spark.sql(
        f"""
        SELECT a.job_id, a.rank_position, a.tier, a.fit_score, a.cv_pdf_path, a.cover_letter_pdf_path,
               a.cover_letter_text, e.title_en, e.company, e.url, s.city, s.country,
               m.reasons, m.matching_skills, m.missing_skills
        FROM {name(Table.GOLD_APPLICATIONS)} a
        JOIN {name(Table.SILVER_JOB_EMBEDDINGS)} e ON e.job_id = a.job_id
        JOIN {name(Table.SILVER_JOBS)} s ON s.job_id = a.job_id
        LEFT JOIN {name(Table.GOLD_JOB_MATCHES)} m ON m.job_id = a.job_id AND m.cv_version = a.cv_version
        WHERE startswith(a.cv_pdf_path, :prefix)
        ORDER BY a.rank_position, a.created_at
        """,
        args={"prefix": day_dir + "/"},
    ).collect()
    if not rows:
        print(f"No applications in {day_dir}; no page written")
        return
    entries = [
        html_export.ApplicationEntry(
            rank=r.rank_position,
            tier=r.tier,
            fit_score=r.fit_score,
            job_id=r.job_id,
            title=r.title_en,
            company=r.company,
            location=", ".join(p for p in (r.city, r.country) if p) or None,
            apply_url=r.url,
            cv_file=html_export.relative_to_day(r.cv_pdf_path, day_dir),
            cover_letter_file=html_export.relative_to_day(r.cover_letter_pdf_path, day_dir),
            cover_letter_text=r.cover_letter_text,
            reasons=r.reasons or "",
            matching_skills=list(r.matching_skills or []),
            missing_skills=list(r.missing_skills or []),
        )
        for r in rows
    ]
    with open(f"{day_dir}/{html_export.JSON_FILE}", "w", encoding="utf-8") as f:
        f.write(html_export.to_json(day, entries))
    with open(f"{day_dir}/{html_export.INDEX_FILE}", "w", encoding="utf-8") as f:
        f.write(html_export.render_index(day, entries))
    print(f"Wrote {day_dir}/{html_export.INDEX_FILE} with {len(entries)} application(s)")


def _create_application(source: CvDocument, cv_version: str, position: int, ranked_job: ranking.RankedJob,
                        job: dict, closing: str, args) -> Application:
    tailored = llm.chat_json(tailoring.TAILOR_SYSTEM_PROMPT, tailoring.tailor_prompt(source, job),
                             tailoring.TailoredCv, model=args.model, max_tokens=12000, temperature=0,
                             context={"source": source})
    letter = llm.chat_json(tailoring.LETTER_SYSTEM_PROMPT, tailoring.letter_prompt(source, tailored, job, closing),
                           tailoring.CoverLetter, model=args.model, max_tokens=4000,
                           context={"source": source, "job": job, "closing": closing})

    today = datetime.now(UTC).date()
    document = tailoring.compose_letter(source, letter, closing, today)
    cv_pdf, tailored = pdf_render.render_cv_within_pages(source, tailored)
    letter_pdf = pdf_render.render_cover_letter(document)

    folder = pdf_render.application_dir(args.output_dir, today, position, ranked_job.company, ranked_job.title_en,
                                        ranked_job.job_id)
    os.makedirs(folder, exist_ok=True)
    cv_path, letter_path = f"{folder}/cv.pdf", f"{folder}/cover_letter.pdf"
    with open(cv_path, "wb") as f:
        f.write(cv_pdf)
    with open(letter_path, "wb") as f:
        f.write(letter_pdf)

    return Application(
        job_id=ranked_job.job_id,
        cv_version=cv_version,
        rank_position=position,
        tier=ranking.tier(ranked_job, args.priority_min_fit),
        fit_score=ranked_job.fit_score,
        tailored_cv_json=tailored.model_dump_json(),
        cover_letter_text=document.text(),
        cv_pdf_path=cv_path,
        cover_letter_pdf_path=letter_path,
        model=args.model,
        created_at=datetime.now(UTC),
    )


def _job_details(spark, name, job_ids: list[str], cv_version: str) -> dict[str, dict]:
    rows = spark.sql(
        f"""
        SELECT s.job_id, s.title_en, s.country, s.city, s.description_en, e.company,
               m.matching_skills, m.missing_skills
        FROM {name(Table.SILVER_JOBS)} s
        JOIN {name(Table.SILVER_JOB_EMBEDDINGS)} e ON e.job_id = s.job_id
        LEFT JOIN {name(Table.GOLD_JOB_MATCHES)} m ON m.job_id = s.job_id AND m.cv_version = :cv_version
        WHERE array_contains(split(:job_ids, ','), s.job_id)
        """,
        args={"job_ids": ",".join(job_ids), "cv_version": cv_version},
    ).collect()
    return {r.job_id: r.asDict() for r in rows}


if __name__ == "__main__":
    main()
