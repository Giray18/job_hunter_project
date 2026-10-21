"""Local console preview of LinkedIn job alert emails.

Usage:
    uv run python -m job_hunter.gmail_preview --max 5
    uv run python -m job_hunter.gmail_preview --query "from:jobalerts-noreply@linkedin.com newer_than:3d" --body
    uv run python -m job_hunter.gmail_preview --max 1 --descriptions
"""

import argparse
import time
from pathlib import Path

import requests

from job_hunter import gmail_client, linkedin_jobs
from job_hunter.records import JobPosting

SECRETS_DIR = Path(".secrets")


def main():
    parser = argparse.ArgumentParser(description="Print matching Gmail messages to the console")
    parser.add_argument("--query", default=gmail_client.LINKEDIN_JOB_ALERTS_QUERY)
    parser.add_argument("--max", type=int, default=5)
    parser.add_argument("--body", action="store_true", help="Print the full message text")
    parser.add_argument("--descriptions", action="store_true", help="Fetch each job's 'About the job' section")
    parser.add_argument("--chars", type=int, default=600, help="Description characters to print (0 = all)")
    parser.add_argument("--client-secret", type=Path, default=SECRETS_DIR / "gmail_client_secret.json")
    parser.add_argument("--token", type=Path, default=SECRETS_DIR / "gmail_token.json")
    args = parser.parse_args()

    creds = gmail_client.credentials_from_local_files(args.client_secret, args.token)
    service = gmail_client.build_service(creds)

    http = requests.Session()

    ids = gmail_client.list_message_ids(service, args.query, args.max)
    print(f"{len(ids)} message(s) for query: {args.query}\n")

    for message_id in ids:
        email = gmail_client.get_message(service, message_id)
        print("=" * 100)
        print(f"Date:    {email.received_at}")
        print(f"From:    {email.sender}")
        print(f"Subject: {email.subject}")
        print(f"Snippet: {email.snippet}")
        print(f"Jobs ({len(email.job_cards)}):")
        for card in email.job_cards:
            print(f"  - {card.title} | {card.company} | {card.location}")
            print(f"    {card.url}")
            if args.descriptions:
                _print_description(linkedin_jobs.fetch_job_posting(card.job_id, http), args.chars)
                time.sleep(2)  # stay polite to LinkedIn's public endpoint
        if args.body:
            print("-" * 100)
            print(email.text)
        print()


def _print_description(posting: JobPosting, chars: int):
    if not posting.ok or not posting.description:
        print(f"    [no description, HTTP {posting.http_status}]")
        return
    for key, value in (posting.criteria or {}).items():
        print(f"    {key}: {value}")
    text = posting.description if chars == 0 else posting.description[:chars] + "…"
    print("    " + text.replace("\n", "\n    "))
    print()


if __name__ == "__main__":
    main()
