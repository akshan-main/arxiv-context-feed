"""Parse backfill workflow inputs from dispatch or GitHub Issue.

Reads from environment variables (set by the workflow) and outputs
the CLI command to run via GITHUB_OUTPUT.

All inputs are validated against strict patterns to prevent injection.
"""

import json
import os
import re
import subprocess
import sys

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9._/:@()+%-]+$")


def get_env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def validate_date(value: str, field: str) -> str:
    """Validate a date string is YYYY-MM-DD format."""
    if not DATE_RE.match(value):
        print(f"Invalid {field}: '{value}' (expected YYYY-MM-DD)", file=sys.stderr)
        sys.exit(1)
    return value


def validate_identifier(value: str) -> str:
    """Validate an identifier contains only safe characters."""
    if value.startswith("-"):
        print(f"Invalid identifier: '{value}' (must not start with -)", file=sys.stderr)
        sys.exit(1)
    if not IDENTIFIER_RE.match(value):
        print(f"Invalid identifier: '{value}' (contains unsafe characters)", file=sys.stderr)
        sys.exit(1)
    return value


def parse_issue_payload(issue_number: str) -> dict:
    """Fetch issue body and extract JSON payload."""
    if not issue_number.isdigit():
        print(f"Invalid issue number: '{issue_number}'", file=sys.stderr)
        sys.exit(1)

    result = subprocess.run(
        ["gh", "issue", "view", issue_number, "--json", "body,labels,state"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Error fetching issue: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(result.stdout)

    if data.get("state", "").upper() == "CLOSED":
        print(f"Issue #{issue_number} is already closed. Skipping.", file=sys.stderr)
        sys.exit(1)

    labels = [l["name"] for l in data.get("labels", [])]
    if "backfill" not in labels:
        print(f"Issue #{issue_number} missing 'backfill' label. Labels: {labels}", file=sys.stderr)
        sys.exit(1)

    body = data.get("body", "")
    match = re.search(r"```json\s*\n(.*?)\n```", body, re.DOTALL)
    if not match:
        print("No JSON payload found in issue body", file=sys.stderr)
        sys.exit(1)

    return json.loads(match.group(1))


def build_top_n_flags(payload: dict) -> str:
    """Build --top-n and --top-n-granularity flags if present."""
    top_n = payload.get("top_n", 0)
    granularity = payload.get("top_n_granularity", "")
    if not top_n or top_n <= 0:
        return ""
    flags = f" --top-n {int(top_n)}"
    if granularity in ("month", "year"):
        flags += f" --top-n-granularity {granularity}"
    return flags


def build_command(payload: dict) -> str:
    """Build CLI command from validated payload. Returns shell-safe command."""
    mode = payload.get("request_type", "single_date")
    dry_run = payload.get("dry_run", False)
    dry_flag = " --dry-run" if dry_run else ""
    top_n_flags = build_top_n_flags(payload)

    if mode == "single_date":
        date = validate_date(payload.get("date", ""), "date")
        return f"contextual-arxiv-feed backfill-date --date {date}{top_n_flags}{dry_flag}"

    elif mode == "date_range":
        start = validate_date(payload.get("start_date", ""), "start_date")
        end = validate_date(payload.get("end_date", ""), "end_date")
        return f"contextual-arxiv-feed backfill --start {start} --end {end}{top_n_flags}{dry_flag}"

    elif mode == "identifiers":
        ids = payload.get("identifiers", [])
        if isinstance(ids, str):
            ids = [i.strip() for i in ids.split(",") if i.strip()]
        if not ids:
            print("No identifiers provided", file=sys.stderr)
            sys.exit(1)
        safe_ids = [validate_identifier(i) for i in ids]
        id_flags = " ".join(f"-i '{ident}'" for ident in safe_ids)
        return f"contextual-arxiv-feed backfill-identifiers {id_flags}{dry_flag}"

    else:
        print(f"Unknown mode: '{mode}'", file=sys.stderr)
        sys.exit(1)


def main():
    issue_number = get_env("INPUT_ISSUE_NUMBER")

    if issue_number:
        print(f"Reading payload from issue #{issue_number}")
        payload = parse_issue_payload(issue_number)
    else:
        mode = get_env("INPUT_MODE", "single_date")
        identifiers_raw = get_env("INPUT_IDENTIFIERS")
        identifiers = [i.strip() for i in identifiers_raw.split(",") if i.strip()] if identifiers_raw else []

        payload = {
            "request_type": mode,
            "date": get_env("INPUT_DATE"),
            "start_date": get_env("INPUT_START_DATE"),
            "end_date": get_env("INPUT_END_DATE"),
            "identifiers": identifiers,
            "dry_run": get_env("INPUT_DRY_RUN").lower() == "true",
        }

    command = build_command(payload)
    print(f"Command: {command}")

    output_file = os.getenv("GITHUB_OUTPUT", "")
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"command={command}\n")
            if issue_number:
                f.write(f"issue_number={issue_number}\n")
    else:
        print(f"GITHUB_OUTPUT not set (local run). Command: {command}")


if __name__ == "__main__":
    main()
