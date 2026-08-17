"""Local file handling, data URIs, and result downloads."""

from __future__ import annotations

import base64
import json
import mimetypes
import re
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from .errors import DownloadError, FileReadError, FileWriteError, SerializationError


@dataclass(frozen=True)
class ResultDownload:
    """URL and fallback extension inferred from the API result metadata."""

    url: str
    fallback_ext: str


def is_url(value: str) -> bool:
    """Return whether value is an HTTP(S) URL."""

    return value.startswith(("http://", "https://"))


def media_value(value: str) -> str:
    """Return URL/data URI or encode a local file as data URI."""

    if is_url(value) or value.startswith("data:"):
        return value
    return file_to_data_uri(Path(value))


def file_to_data_uri(path: Path) -> str:
    """Encode a local file as data URI for JSON API endpoints."""

    if not path.exists() or not path.is_file():
        raise FileReadError(
            "Unable to read input file",
            code="file_not_found",
            details={"path": str(path)},
        )
    mime, _ = mimetypes.guess_type(path.name)
    if not mime:
        mime = "application/octet-stream"
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError as exc:
        raise FileReadError(
            "Unable to read input file", details={"path": str(path)}
        ) from exc
    return f"data:{mime};base64,{encoded}"


def result_urls(value: Any) -> list[str]:
    """Recursively collect URLs from an arbitrary API result."""

    urls: list[str] = []
    if isinstance(value, str) and is_url(value):
        urls.append(value)
    elif isinstance(value, list):
        for item in value:
            urls.extend(result_urls(item))
    elif isinstance(value, dict):
        for item in value.values():
            urls.extend(result_urls(item))
    return urls


def result_downloads(value: Any, *, fallback_ext: str = ".bin") -> list[ResultDownload]:
    """Recursively collect downloadable URLs with metadata-based fallback extensions."""

    downloads: list[ResultDownload] = []
    if isinstance(value, str) and is_url(value):
        downloads.append(ResultDownload(value, fallback_ext))
    elif isinstance(value, list):
        for item in value:
            downloads.extend(result_downloads(item, fallback_ext=fallback_ext))
    elif isinstance(value, dict):
        scoped_fallback = extension_from_metadata(value, fallback_ext)
        for item in value.values():
            downloads.extend(result_downloads(item, fallback_ext=scoped_fallback))
    return downloads


def extension_from_metadata(value: dict[str, Any], fallback: str) -> str:
    """Infer a file extension from common result metadata fields."""

    for key in ("mime_type", "content_type", "contentType", "mime"):
        raw = value.get(key)
        if isinstance(raw, str):
            ext = extension_for_mime(raw)
            if ext:
                return ext
    for key in ("extension", "ext", "format", "file_format", "type"):
        raw = value.get(key)
        if isinstance(raw, str):
            ext = normalize_extension(raw)
            if ext:
                return ext
    return normalize_extension(fallback) or ".bin"


def extension_for_mime(content_type: str | None) -> str | None:
    """Return an extension for a MIME type."""

    if not content_type:
        return None
    ext = mimetypes.guess_extension(content_type.split(";")[0].strip())
    return normalize_extension(ext) if ext else None


def normalize_extension(value: str | None) -> str | None:
    """Normalize extension-like API data to a safe suffix."""

    if not value:
        return None
    candidate = value.strip().lower()
    if "/" in candidate:
        return extension_for_mime(candidate)
    if not candidate.startswith("."):
        candidate = f".{candidate}"
    if re.fullmatch(r"\.[a-z0-9][a-z0-9]{0,9}", candidate):
        return candidate
    return None


def extension_for(url: str, content_type: str | None, fallback: str) -> str:
    """Infer a result extension from Content-Type or URL."""

    ext = extension_for_mime(content_type)
    if ext:
        return ext
    suffix = Path(urlparse(url).path).suffix
    return normalize_extension(suffix) or normalize_extension(fallback) or ".bin"


def safe_task_stem(task_id: str | None) -> str | None:
    """Return a filesystem-safe task id stem without path traversal."""

    if not task_id:
        return None
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", task_id).strip("._")
    return stem or None


def result_target(
    output_dir: Path, *, task_id: str | None, index: int, total: int, ext: str
) -> Path:
    """Build a target path for a downloaded or inline result."""

    stem = safe_task_stem(task_id)
    if stem:
        name = f"{stem}{ext}" if total == 1 else f"{stem}-{index}{ext}"
    else:
        name = f"result-{index}{ext}"
    return output_dir / name


def download_urls(
    urls: list[str] | list[ResultDownload],
    output_dir: Path,
    *,
    fallback_ext: str = ".bin",
    task_id: str | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> list[str]:
    """Download URLs into a directory and return saved file paths."""

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise FileWriteError(
            "Unable to create download directory", details={"path": str(output_dir)}
        ) from exc
    downloaded: list[str] = []
    items = [
        item if isinstance(item, ResultDownload) else ResultDownload(item, fallback_ext)
        for item in urls
    ]
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        total = len(items)
        for index, item in enumerate(items, start=1):
            url = item.url
            target: Path | None = None
            partial: Path | None = None
            try:
                if cancel_check is not None:
                    cancel_check()
                with client.stream("GET", url) as response:
                    response.raise_for_status()
                    ext = extension_for(
                        url,
                        response.headers.get("content-type"),
                        item.fallback_ext,
                    )
                    target = result_target(
                        output_dir,
                        task_id=task_id,
                        index=index,
                        total=total,
                        ext=ext,
                    )
                    partial = target.with_name(f".{target.name}.part")
                    with partial.open("wb") as file_handle:
                        for chunk in response.iter_bytes():
                            if cancel_check is not None:
                                cancel_check()
                            file_handle.write(chunk)
                    if cancel_check is not None:
                        cancel_check()
                    partial.replace(target)
                    downloaded.append(str(target))
            except httpx.HTTPError as exc:
                raise DownloadError(
                    "Unable to download result", details={"url": url.split("?")[0]}
                ) from exc
            except OSError as exc:
                raise FileWriteError(
                    "Unable to write downloaded result",
                    details={"path": str(target or output_dir)},
                ) from exc
            finally:
                if partial is not None:
                    with suppress(OSError):
                        partial.unlink(missing_ok=True)
    return downloaded


def save_inline_result(
    value: Any,
    output_dir: Path,
    *,
    task_id: str,
    fallback_ext: str = ".bin",
    cancel_check: Callable[[], None] | None = None,
) -> list[str]:
    """Save non-URL API results such as ASR text or structured transcription JSON."""

    if value is None:
        return []
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise FileWriteError(
            "Unable to create download directory", details={"path": str(output_dir)}
        ) from exc
    ext, content, encoding = inline_result_payload(value, fallback_ext=fallback_ext)
    target = result_target(output_dir, task_id=task_id, index=1, total=1, ext=ext)
    try:
        if cancel_check is not None:
            cancel_check()
        if encoding:
            target.write_text(content, encoding=encoding)
        else:
            target.write_bytes(content)
    except OSError as exc:
        raise FileWriteError(
            "Unable to write result file", details={"path": str(target)}
        ) from exc
    return [str(target)]


def inline_result_payload(
    value: Any, *, fallback_ext: str
) -> tuple[str, Any, str | None]:
    """Return extension, content and text encoding for a non-URL result."""

    fallback = normalize_extension(fallback_ext) or ".bin"
    if isinstance(value, str):
        return ".txt" if fallback == ".bin" else fallback, value, "utf-8"
    if is_plain_transcript(value):
        transcript = next(item for item in value.values() if isinstance(item, str))
        return ".txt", transcript, "utf-8"
    if isinstance(value, (dict, list)):
        try:
            return ".json", json.dumps(value, ensure_ascii=False, indent=2), "utf-8"
        except (TypeError, ValueError) as exc:
            raise SerializationError("Unable to serialize result for saving") from exc
    return fallback, str(value), "utf-8"


def is_plain_transcript(value: Any) -> bool:
    """Return whether an ASR result is only a single transcript string."""

    if not isinstance(value, dict) or len(value) != 1:
        return False
    key, item = next(iter(value.items()))
    return key in {"text", "transcript", "transcription"} and isinstance(item, str)
