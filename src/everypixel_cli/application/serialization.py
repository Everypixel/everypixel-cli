"""Convert application results into the public CLI/MCP-safe data contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .models import OperationResult


def serialize_operation_result(result: OperationResult) -> Any:
    """Preserve the existing CLI success payload shape."""

    value = result.value
    if isinstance(value, Mapping):
        payload = dict(value)
        if result.saved_files:
            payload["downloaded"] = [str(path) for path in result.saved_files]
        return payload
    return value
