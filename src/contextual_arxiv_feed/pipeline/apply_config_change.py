"""Apply configuration changes from GitHub Issues.

Processes issues created by Google Form bot:
- Validates payload structure
- Validates topic key regex, category allowlist
- Validates judge changes are whitelisted
- Updates YAML files deterministically
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from contextual_arxiv_feed.config import (
    TOPIC_KEY_PATTERN,
    CategoriesConfig,
    JudgeConfig,
    load_yaml,
)

logger = logging.getLogger(__name__)

REQUIRED_LABELS = {"source:google-form", "config-change"}


@dataclass
class ValidationError:
    """A single validation error."""

    field: str
    message: str


@dataclass
class ChangeResult:
    """Result of applying a config change."""

    success: bool
    change_type: str = ""
    target_type: str = ""
    errors: list[ValidationError] = field(default_factory=list)
    changes_made: list[str] = field(default_factory=list)

    def add_error(self, field: str, message: str) -> None:
        """Add a validation error."""
        self.errors.append(ValidationError(field=field, message=message))
        self.success = False


class IssuePayload:
    """Parsed and validated issue payload."""

    def __init__(self, raw_payload: dict[str, Any]):
        """Initialize with raw payload.

        Args:
            raw_payload: Raw JSON payload from issue body.
        """
        self._raw = raw_payload
        self.target_repo = raw_payload.get("target_repo", "")
        self.change_type: Literal["add", "update", "remove"] = raw_payload.get("change_type", "")
        self.target_type: Literal["topic", "judge"] = raw_payload.get("target_type", "")
        self.topic_data = raw_payload.get("topic", {})
        self.judge_data = raw_payload.get("judge", {})


def parse_issue_payload(issue_body: str) -> IssuePayload | None:
    """Parse JSON payload from issue body.

    Args:
        issue_body: Raw issue body text.

    Returns:
        IssuePayload or None if parsing fails.
    """
    try:
        json_match = re.search(r"```json\s*(.*?)\s*```", issue_body, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r"\{.*\}", issue_body, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                logger.error("No JSON found in issue body")
                return None

        data = json.loads(json_str)
        return IssuePayload(data)

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return None


def validate_payload(payload: IssuePayload, categories: CategoriesConfig) -> ChangeResult:
    """Validate the payload structure and content.

    Args:
        payload: Parsed payload.
        categories: Valid categories config.

    Returns:
        ChangeResult with validation errors if any.
    """
    result = ChangeResult(
        success=True,
        change_type=payload.change_type,
        target_type=payload.target_type,
    )

    if payload.target_repo != "contextual-arxiv-feed":
        result.add_error("target_repo", f"Expected 'contextual-arxiv-feed', got '{payload.target_repo}'")

    if payload.change_type not in ("add", "update", "remove"):
        result.add_error("change_type", f"Must be 'add', 'update', or 'remove', got '{payload.change_type}'")

    if payload.target_type not in ("topic", "judge"):
        result.add_error("target_type", f"Must be 'topic' or 'judge', got '{payload.target_type}'")

    if payload.target_type == "topic":
        _validate_topic_payload(payload, categories, result)
    elif payload.target_type == "judge":
        _validate_judge_payload(payload, result)

    return result


def _validate_topic_payload(
    payload: IssuePayload, categories: CategoriesConfig, result: ChangeResult
) -> None:
    """Validate topic change payload.

    Args:
        payload: The payload.
        categories: Valid categories.
        result: Result to add errors to.
    """
    topic = payload.topic_data

    if not topic:
        result.add_error("topic", "Topic data is required")
        return

    key = topic.get("key", "")
    if not key:
        result.add_error("topic.key", "Topic key is required")
    elif not TOPIC_KEY_PATTERN.match(key):
        result.add_error(
            "topic.key",
            f"Key '{key}' must match pattern: lowercase alphanumeric with hyphens/underscores",
        )

    if payload.change_type in ("add", "update"):
        if not topic.get("name"):
            result.add_error("topic.name", "Topic name is required")

        arxiv_categories = topic.get("arxiv_categories", [])
        for cat in arxiv_categories:
            if not categories.is_valid(cat):
                result.add_error(
                    "topic.arxiv_categories",
                    f"Invalid arXiv category '{cat}'",
                )

        keywords = topic.get("keywords", [])
        phrases = topic.get("phrases", [])
        if not keywords and not phrases:
            result.add_error(
                "topic.keywords",
                "At least one keyword or phrase is required",
            )


def _validate_judge_payload(payload: IssuePayload, result: ChangeResult) -> None:
    """Validate judge change payload.

    Args:
        payload: The payload.
        result: Result to add errors to.
    """
    judge = payload.judge_data

    if not judge:
        result.add_error("judge", "Judge data is required")
        return

    # Only whitelisted fields can be edited
    config = JudgeConfig()
    allowed_fields = set(config.editable_fields)

    for key in judge:
        if key not in allowed_fields:
            result.add_error(
                f"judge.{key}",
                f"Field '{key}' is not editable. Allowed: {allowed_fields}",
            )

    if "strictness" in judge and judge["strictness"] not in ("low", "medium", "high"):
        result.add_error(
            "judge.strictness",
            f"Must be 'low', 'medium', or 'high', got '{judge['strictness']}'",
        )


def apply_topic_change(
    payload: IssuePayload,
    config_dir: Path,
) -> ChangeResult:
    """Apply a topic configuration change.

    Args:
        payload: Validated payload.
        config_dir: Path to config directory.

    Returns:
        ChangeResult with changes made.
    """
    result = ChangeResult(success=True, change_type=payload.change_type, target_type="topic")

    topics_path = config_dir / "topics.yaml"
    data = load_yaml(topics_path)

    topics = data.get("topics", [])
    topic_key = payload.topic_data.get("key")

    if payload.change_type == "add":
        for t in topics:
            if t.get("key") == topic_key:
                result.add_error("topic.key", f"Topic '{topic_key}' already exists")
                return result

        topics.append(payload.topic_data)
        result.changes_made.append(f"Added topic '{topic_key}'")

    elif payload.change_type == "update":
        found = False
        for i, t in enumerate(topics):
            if t.get("key") == topic_key:
                topics[i] = {**t, **payload.topic_data}
                found = True
                result.changes_made.append(f"Updated topic '{topic_key}'")
                break

        if not found:
            result.add_error("topic.key", f"Topic '{topic_key}' not found")
            return result

    elif payload.change_type == "remove":
        found = False
        for i, t in enumerate(topics):
            if t.get("key") == topic_key:
                topics.pop(i)
                found = True
                result.changes_made.append(f"Removed topic '{topic_key}'")
                break

        if not found:
            result.add_error("topic.key", f"Topic '{topic_key}' not found")
            return result

    data["topics"] = topics
    with open(topics_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    return result


def apply_judge_change(
    payload: IssuePayload,
    config_dir: Path,
) -> ChangeResult:
    """Apply a judge configuration change.

    Args:
        payload: Validated payload.
        config_dir: Path to config directory.

    Returns:
        ChangeResult with changes made.
    """
    result = ChangeResult(success=True, change_type=payload.change_type, target_type="judge")

    judge_path = config_dir / "judge.yaml"
    data = load_yaml(judge_path)

    for key, value in payload.judge_data.items():
        old_value = data.get(key)
        data[key] = value
        result.changes_made.append(f"Updated judge.{key}: {old_value} -> {value}")

    if result.changes_made:
        data["prompt_version"] = data.get("prompt_version", 1) + 1
        result.changes_made.append(f"Bumped prompt_version to {data['prompt_version']}")

    with open(judge_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    return result


def apply_config_change(
    issue_body: str,
    config_dir: Path,
    categories: CategoriesConfig | None = None,
) -> ChangeResult:
    """Apply a configuration change from issue body.

    Main entry point for the config change handler.

    Args:
        issue_body: Raw issue body text.
        config_dir: Path to config directory.
        categories: Categories config (loads from file if None).

    Returns:
        ChangeResult with success/failure and changes.
    """
    payload = parse_issue_payload(issue_body)
    if not payload:
        return ChangeResult(
            success=False,
            errors=[ValidationError(field="payload", message="Failed to parse JSON payload")],
        )

    if categories is None:
        categories_data = load_yaml(config_dir / "categories.yaml")
        categories = CategoriesConfig(**categories_data)

    validation = validate_payload(payload, categories)
    if not validation.success:
        return validation

    if payload.target_type == "topic":
        return apply_topic_change(payload, config_dir)
    elif payload.target_type == "judge":
        return apply_judge_change(payload, config_dir)
    else:
        return ChangeResult(
            success=False,
            errors=[ValidationError(field="target_type", message=f"Unknown type: {payload.target_type}")],
        )
