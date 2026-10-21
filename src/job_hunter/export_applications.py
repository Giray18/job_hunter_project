"""Local command: copy one day's applications from the UC volume to Downloads and build the local page.

Usage:
    uv run python -m job_hunter.export_applications                      # today, prod schema
    uv run python -m job_hunter.export_applications --date 2026-10-21
    uv run python -m job_hunter.export_applications --schema dev_<you>_medallion --profile personal-projects

The page is re-rendered with absolute local file paths, so a browser agent (e.g. Claude in Chrome) can upload
the CV and cover letter from disk.
"""

import argparse
import os
from datetime import date
from pathlib import Path

from databricks.sdk import WorkspaceClient

from job_hunter import html_export, pdf_render

DEFAULT_DEST = Path.home() / "Downloads" / "JobApplications"


def main():
    parser = argparse.ArgumentParser(description="Download a day's job applications and build the local page")
    parser.add_argument("--date", type=date.fromisoformat, default=date.today())
    parser.add_argument("--catalog", default="job_hunter_project")
    parser.add_argument("--schema", default="medallion", help="prod: medallion; dev: dev_<short_name>_medallion")
    parser.add_argument("--volume", default="cv")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--profile", default=os.environ.get("DATABRICKS_CONFIG_PROFILE"))
    args = parser.parse_args()

    client = WorkspaceClient(profile=args.profile) if args.profile else WorkspaceClient()
    remote = pdf_render.day_dir(f"/Volumes/{args.catalog}/{args.schema}/{args.volume}/output", args.date)
    local = args.dest / args.date.isoformat()

    count = download_tree(client, remote, local)
    json_path = local / html_export.JSON_FILE
    if not json_path.exists():
        raise SystemExit(f"No {html_export.JSON_FILE} in {remote}; did tailor_applications run on {args.date}?")

    day, entries = html_export.from_json(json_path.read_text(encoding="utf-8"))
    index = local / html_export.INDEX_FILE
    index.write_text(html_export.render_index(day, entries, local_dir=local), encoding="utf-8")
    print(f"Downloaded {count} file(s) to {local}")
    print(f"{len(entries)} application(s). Open: {index.resolve().as_uri()}")


def download_tree(client: WorkspaceClient, remote_dir: str, local_dir: Path) -> int:
    """Recursively copy a volume directory; returns the number of files written."""
    local_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for entry in client.files.list_directory_contents(remote_dir):
        target = local_dir / entry.name
        if entry.is_directory:
            count += download_tree(client, entry.path, target)
        else:
            with client.files.download(entry.path).contents as src:
                target.write_bytes(src.read())
            count += 1
    return count


if __name__ == "__main__":
    main()
