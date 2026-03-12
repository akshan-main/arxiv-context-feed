"""Streamlit app for creating backfill requests via GitHub Issues.

This app ONLY creates GitHub issues — it does NOT run ingestion.
The backfill.yml workflow picks up the issue and runs the actual pipeline.

Usage:
    streamlit run streamlit_backfill/app.py

Requires env vars:
    GITHUB_TOKEN  - Personal access token with repo scope
    GITHUB_REPO   - owner/repo format
"""

import json
import os
import re
from datetime import date, timedelta
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO = os.getenv("GITHUB_REPO", "")

ARXIV_ID_RE = re.compile(r"\d{4}\.\d{4,5}(v\d+)?$")
ARXIV_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/\d{4}\.\d{4,5}")
DOI_RE = re.compile(r"^10\.\d{4,}/\S+$")


def validate_identifier(ident: str) -> tuple[bool, str]:
    """Check if identifier looks like a valid arXiv ID, DOI, or arXiv URL."""
    ident = ident.strip()
    if ARXIV_ID_RE.match(ident):
        return True, "arXiv ID"
    if ARXIV_URL_RE.search(ident):
        return True, "arXiv URL"
    if DOI_RE.match(ident):
        return True, "DOI"
    return False, "unknown"


def preview_identifiers(identifiers: list[str]) -> None:
    """Show a preview table of parsed identifiers."""
    rows = []
    for ident in identifiers:
        valid, id_type = validate_identifier(ident)
        rows.append({
            "Identifier": ident,
            "Type": id_type,
            "Valid": "yes" if valid else "no",
        })
    st.table(rows)


def create_issue(title: str, body: str, labels: list[str]) -> dict | None:
    """Create a GitHub issue. Returns issue data or None on failure."""
    resp = requests.post(
        f"https://api.github.com/repos/{GITHUB_REPO}/issues",
        headers={
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
        },
        json={"title": title, "body": body, "labels": labels},
        timeout=30,
    )
    if resp.status_code == 201:
        return resp.json()
    st.error(f"GitHub API error {resp.status_code}: {resp.text}")
    return None


st.set_page_config(page_title="arXiv Backfill Request", layout="centered")
st.title("arXiv Backfill Request")
st.caption("Creates a GitHub Issue to trigger the backfill workflow. No ingestion happens here.")

if not GITHUB_TOKEN or not GITHUB_REPO:
    st.error(
        "Set `GITHUB_TOKEN` and `GITHUB_REPO` env vars before running.\n\n"
        "```bash\n"
        "export GITHUB_TOKEN=ghp_...\n"
        "export GITHUB_REPO=owner/repo\n"
        "streamlit run streamlit_backfill/app.py\n"
        "```"
    )
    st.stop()

mode = st.radio(
    "Request mode",
    ["single_date", "date_range", "identifiers"],
    format_func=lambda x: {
        "single_date": "Single Date",
        "date_range": "Date Range",
        "identifiers": "Identifiers (arXiv ID / DOI / URL)",
    }[x],
)

payload: dict = {
    "request_type": mode,
    "date": "",
    "start_date": "",
    "end_date": "",
    "identifiers": [],
    "dry_run": False,
    "requested_by": "streamlit",
    "note": "",
}

can_submit = True

if mode == "single_date":
    d = st.date_input("Date", value=date.today() - timedelta(days=1))
    payload["date"] = d.isoformat()

elif mode == "date_range":
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Start date", value=date.today() - timedelta(days=7))
    with col2:
        end = st.date_input("End date", value=date.today() - timedelta(days=1))
    if start > end:
        st.error("Start date must be before or equal to end date.")
        can_submit = False
    payload["start_date"] = start.isoformat()
    payload["end_date"] = end.isoformat()

elif mode == "identifiers":
    raw = st.text_area(
        "Identifiers (one per line)",
        placeholder="2401.12345\n10.48550/arXiv.2401.12345\nhttps://arxiv.org/abs/2401.12345",
        height=150,
    )
    ids = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    payload["identifiers"] = ids

    if ids:
        st.subheader("Identifier preview")
        preview_identifiers(ids)
        invalid = [i for i in ids if not validate_identifier(i)[0]]
        if invalid:
            st.warning(f"{len(invalid)} identifier(s) could not be validated: {invalid}")
    else:
        st.info("Paste at least one identifier above.")
        can_submit = False

payload["dry_run"] = st.checkbox("Dry run (no actual ingestion)", value=False)
payload["note"] = st.text_input("Note (optional)", placeholder="requested historical ingest")

st.subheader("Payload")
st.json(payload)

if st.button("Create GitHub Issue", type="primary", disabled=not can_submit):
    today = date.today().isoformat()
    title_map = {
        "single_date": f"[Backfill] single_date - {payload['date']}",
        "date_range": f"[Backfill] date_range - {payload['start_date']} to {payload['end_date']}",
        "identifiers": f"[Backfill] identifiers - {today}",
    }
    title = title_map[mode]
    if payload["dry_run"]:
        title += " [DRY RUN]"

    body = (
        "## Backfill Request\n\n"
        f"**Mode:** `{mode}`\n"
        f"**Dry run:** `{payload['dry_run']}`\n"
    )
    if payload["note"]:
        body += f"**Note:** {payload['note']}\n"
    body += (
        "\n### Payload\n\n"
        f"```json\n{json.dumps(payload, indent=2)}\n```\n\n"
        "---\n*Created via Streamlit backfill app*\n"
    )

    with st.spinner("Creating issue..."):
        issue = create_issue(title, body, ["backfill", "streamlit-request"])

    if issue:
        url = issue["html_url"]
        num = issue["number"]
        st.success(f"Issue #{num} created!")
        st.markdown(f"[Open issue #{num}]({url})")
