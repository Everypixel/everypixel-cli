"""Typed request and result models used independently from the CLI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Any, Mapping


class OperationCancelled(Exception):
    """Internal control flow raised when a caller abandons an operation."""


@dataclass(frozen=True)
class ExecutionOptions:
    """Execution concerns shared by synchronous and asynchronous operations."""

    wait: bool = False
    download_directory: Path | None = None
    timeout: float = 1800.0
    poll_interval: float = 2.0
    cancel_event: Event | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @property
    def wait_for_result(self) -> bool:
        """Downloads always require a completed task."""

        return self.wait or self.download_directory is not None

    def check_cancelled(self) -> None:
        """Raise when the caller no longer wants this operation to continue."""

        if self.cancel_event is not None and self.cancel_event.is_set():
            raise OperationCancelled


@dataclass(frozen=True)
class OperationRequest:
    """Transport-independent API operation request."""

    endpoint: str
    payload: Mapping[str, Any]
    method: str = "POST"
    execution: ExecutionOptions = ExecutionOptions()
    fallback_extension: str = ".bin"


@dataclass(frozen=True)
class OperationResult:
    """Application result that contains no presentation or HTTP objects."""

    value: Any
    task: Mapping[str, Any] | None = None
    saved_files: tuple[Path, ...] = ()
