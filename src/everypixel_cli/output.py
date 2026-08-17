"""CLI output rendering in JSON or human-readable mode."""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text

from .errors import CLIError, JqExpressionError, SerializationError, serialize_error


def apply_jq(data: Any, expr: str | None) -> Any:
    """Apply a jq expression or simple dot selector fallback."""

    if not expr:
        return data
    try:
        import jq  # type: ignore
    except ModuleNotFoundError:
        return apply_simple_selector(data, expr)
    try:
        return jq.compile(expr).input(data).first()
    except Exception as exc:  # jq does not expose stable public exception classes.
        raise JqExpressionError(
            "Unable to apply jq expression", details={"expression": expr}
        ) from exc


def apply_simple_selector(data: Any, expr: str) -> Any:
    """Minimal fallback for `.field.subfield` expressions."""

    if not expr.startswith("."):
        raise JqExpressionError(
            "Unable to apply jq expression", details={"expression": expr}
        )
    current = data
    try:
        for part in expr[1:].split("."):
            if not part:
                continue
            if not isinstance(current, dict):
                raise TypeError("selector target is not an object")
            current = current[part]
        return current
    except (KeyError, TypeError) as exc:
        raise JqExpressionError(
            "Unable to apply jq expression", details={"expression": expr}
        ) from exc


def emit_json(data: Any, jq_expr: str | None = None) -> None:
    """Print JSON output."""

    try:
        selected = apply_jq(data, jq_expr)
        print(json.dumps(selected, ensure_ascii=False, indent=2))
    except CLIError:
        raise
    except (TypeError, ValueError) as exc:
        raise SerializationError("Unable to serialize CLI response") from exc


def emit_human(data: Any, *, title: str | None = None, no_color: bool = False) -> None:
    """Print output using Rich tables or a simple fallback."""

    console = Console(no_color=no_color)
    if title:
        console.print(f"[bold]{title}[/bold]")
    if isinstance(data, dict) and render_known_table(console, data):
        return
    if isinstance(data, dict):
        table = Table(show_header=False, box=None)
        table.add_column("Key", style="cyan")
        table.add_column("Value")
        for key, value in data.items():
            table.add_row(
                str(key),
                json.dumps(value, ensure_ascii=False)
                if isinstance(value, (dict, list))
                else str(value),
            )
        console.print(table)
    else:
        console.print(data)


def render_known_table(console: Console, data: dict[str, Any]) -> bool:
    """Render specialized tables for known API responses."""

    if isinstance(data.get("keywords"), list):
        table = Table(title="Keywords")
        table.add_column("Keyword", style="cyan")
        table.add_column("Score", justify="right")
        for item in data["keywords"]:
            if isinstance(item, dict):
                table.add_row(
                    str(item.get("keyword", "")), format_score(item.get("score"))
                )
        console.print(table)
        return True

    if isinstance(data.get("faces"), list):
        table = Table(title="Faces")
        table.add_column("#", justify="right")
        table.add_column("Score", justify="right")
        table.add_column("BBox")
        table.add_column("Age")
        table.add_column("Gender")
        for index, item in enumerate(data["faces"], start=1):
            if isinstance(item, dict):
                table.add_row(
                    str(index),
                    format_score(item.get("score")),
                    json.dumps(item.get("bbox", ""), ensure_ascii=False),
                    str(item.get("age", "")),
                    str(item.get("gender", "")),
                )
        console.print(table)
        return True

    if isinstance(data.get("quality"), dict):
        table = Table(title="Quality")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        for key, value in data["quality"].items():
            table.add_row(
                str(key),
                format_score(value) if isinstance(value, float) else str(value),
            )
        console.print(table)
        return True

    return False


def format_score(value: Any) -> str:
    """Format score/metric values for human output."""

    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return ""
    return str(value)


def emit_error(error: CLIError, *, output_json: bool, no_color: bool = False) -> None:
    """Print a JSON error to stdout or a human error to stderr."""

    if output_json:
        # JSON output has one machine-readable stream, including failures.
        print(serialize_error(error))
        return
    console = Console(stderr=True, no_color=no_color)
    console.print(Text(f"Error [{error.code}]: {error.message}", style="red"))
    for key, value in error.details.items():
        console.print(f"{key.replace('_', ' ').title()}: {value}")
