"""Local morning sync: wait for today's pipeline run, then download the day's applications to Downloads.

Meant to be started by Windows Task Scheduler shortly after the daily Databricks job
(see scripts/register_daily_sync.ps1):
    uv run python -m job_hunter.sync_today --profile personal-projects --open

Polls the job's runs that started today until one finishes; on success (or a failed run that still wrote
applications) it calls export_applications and optionally opens the page. Everything is logged to
~/Downloads/JobApplications/sync.log.
"""

import argparse
import logging
import os
import time
from datetime import date, datetime
from pathlib import Path

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import RunLifeCycleState, RunResultState

from job_hunter.export_applications import DEFAULT_DEST, export

FINISHED = {RunLifeCycleState.TERMINATED, RunLifeCycleState.SKIPPED, RunLifeCycleState.INTERNAL_ERROR}
log = logging.getLogger("job_hunter.sync_today")


def main():
    parser = argparse.ArgumentParser(description="Wait for today's pipeline run and download its applications")
    parser.add_argument("--job-name", default="job_hunter_ingestion", help="Prod job name in the workspace")
    parser.add_argument("--catalog", default="job_hunter_project")
    parser.add_argument("--schema", default="medallion")
    parser.add_argument("--volume", default="cv")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--profile", default=os.environ.get("DATABRICKS_CONFIG_PROFILE"))
    parser.add_argument("--timeout-minutes", type=int, default=180)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--open", action="store_true", help="Open the page in the default browser when done")
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.dest / "sync.log", encoding="utf-8"), logging.StreamHandler()],
    )
    client = WorkspaceClient(profile=args.profile) if args.profile else WorkspaceClient()
    today = date.today()

    run = wait_for_todays_run(client, args.job_name, today, args.timeout_minutes * 60, args.poll_seconds)
    if run is None:
        log.error("No finished run of %s today within %s minutes; nothing downloaded", args.job_name,
                  args.timeout_minutes)
        raise SystemExit(1)
    result = run.state.result_state if run.state else None
    log.info("Run %s finished: %s %s", run.run_id, run.state.life_cycle_state if run.state else None, result)

    try:
        index, entries, count = export(client, args.catalog, args.schema, args.volume, today, args.dest)
    except FileNotFoundError as e:
        # A successful run with no new top jobs writes no page; that is not an error.
        log.warning("%s", e)
        raise SystemExit(0 if result == RunResultState.SUCCESS else 1) from e
    log.info("Downloaded %s file(s); %s application(s): %s", count, len(entries), index.resolve().as_uri())
    if args.open:
        os.startfile(index)  # opens the local page in the default browser (Windows only)


def wait_for_todays_run(client: WorkspaceClient, job_name: str, today: date, timeout_seconds: int,
                        poll_seconds: int):
    """Latest run of `job_name` started today, once it has finished; None on timeout."""
    jobs = list(client.jobs.list(name=job_name))
    if not jobs:
        raise SystemExit(f"Job {job_name!r} not found in the workspace")
    job_id = jobs[0].job_id
    since_ms = int(datetime.combine(today, datetime.min.time()).timestamp() * 1000)
    deadline = time.monotonic() + timeout_seconds
    while True:
        runs = list(client.jobs.list_runs(job_id=job_id, start_time_from=since_ms, limit=5))
        if runs:
            latest = max(runs, key=lambda r: r.start_time or 0)
            state = latest.state.life_cycle_state if latest.state else None
            if state in FINISHED:
                return latest
            log.info("Run %s is %s; waiting %ss", latest.run_id, state, poll_seconds)
        else:
            log.info("No run of %s has started today yet; waiting %ss", job_name, poll_seconds)
        if time.monotonic() + poll_seconds > deadline:
            return None
        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
