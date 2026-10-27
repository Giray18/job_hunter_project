"""Post-deploy guard: the chat app's service principal must hold every Unity Catalog privilege its app resources
declare, plus USE CATALOG / USE SCHEMA on their parents. Missing privileges are granted again and reported.

Run in CI after `bundle deploy` (as the runner, which has MANAGE on the catalog):
    python -m job_hunter.app_grants --app job-hunter-chat
"""

import argparse
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Requirement:
    securable_type: str  # catalog / schema / table / volume (REST path form)
    full_name: str
    privilege: str


def required_privileges(app: dict) -> list[Requirement]:
    """Privileges implied by the app's uc_securable resources (an app as returned by GET /api/2.0/apps/{name})."""
    needed: list[Requirement] = []
    for resource in app.get("resources") or []:
        securable = resource.get("uc_securable")
        if not securable:
            continue
        full_name = securable["securable_full_name"]
        catalog, schema, _ = full_name.split(".", 2)
        needed += [
            Requirement("catalog", catalog, "USE_CATALOG"),
            Requirement("schema", f"{catalog}.{schema}", "USE_SCHEMA"),
            Requirement(securable["securable_type"].lower(), full_name, securable["permission"]),
        ]
    return list(dict.fromkeys(needed))


def effective_privileges(response: dict, principal: str) -> set[str]:
    """Privileges of `principal` in an effective-permissions response, direct or inherited (ALL_PRIVILEGES counts)."""
    held = set()
    for assignment in response.get("privilege_assignments") or []:
        if assignment.get("principal") == principal:
            held |= {p["privilege"] for p in assignment.get("privileges") or []}
    return held


def is_missing(requirement: Requirement, held: set[str]) -> bool:
    return requirement.privilege not in held and "ALL_PRIVILEGES" not in held


def main():
    parser = argparse.ArgumentParser(description="Check and restore the chat app's Unity Catalog privileges")
    parser.add_argument("--app", required=True, help="Databricks App name, e.g. job-hunter-chat")
    parser.add_argument("--check-only", action="store_true", help="Report missing privileges without granting them")
    args = parser.parse_args()

    from databricks.sdk import WorkspaceClient

    api = WorkspaceClient().api_client
    app = api.do("GET", f"/api/2.0/apps/{args.app}")
    principal = app["service_principal_client_id"]
    missing = []
    for req in required_privileges(app):
        response = api.do("GET", f"/api/2.1/unity-catalog/effective-permissions/{req.securable_type}/{req.full_name}",
                          query={"principal": principal})
        held = effective_privileges(response, principal)
        status = "missing" if is_missing(req, held) else "ok"
        print(f"{status:8} {req.privilege} on {req.securable_type} {req.full_name}")
        if status == "missing":
            missing.append(req)

    if not missing:
        print(f"{args.app} ({principal}) holds all {len(required_privileges(app))} required privileges")
        return
    for req in missing:
        # GitHub Actions shows ::warning:: lines on the run summary, so a repair never goes unnoticed.
        print(f"::warning::{args.app} was missing {req.privilege} on {req.securable_type} {req.full_name}"
              + ("" if args.check_only else "; granted again"))
        if not args.check_only:
            api.do("PATCH", f"/api/2.1/unity-catalog/permissions/{req.securable_type}/{req.full_name}",
                   body={"changes": [{"principal": principal, "add": [req.privilege]}]})
    if args.check_only:
        raise SystemExit(json.dumps([vars(r) for r in missing]))


if __name__ == "__main__":
    main()
