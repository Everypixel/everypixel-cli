from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from everypixel_cli.application import ExecutionOptions
from everypixel_cli.application.models import OperationRequest
from everypixel_cli.application.serialization import serialize_operation_result
from everypixel_cli.application.services import (
    ApplicationServices,
    DownloadService,
    OperationService,
    build_video_edit_payload,
)
from everypixel_cli.errors import (
    APIResponseError,
    FileWriteError,
    TaskFailedError,
    TaskPollingError,
)


class FakeClient:
    base_url = "https://api.test"

    def __init__(self, response: Any, statuses: list[Any] | None = None):
        self.response = response
        self.statuses = list(statuses or [])
        self.requests: list[dict[str, Any]] = []

    def request(
        self, method, path, *, json=None, params=None, files=None, auth_required=True
    ):
        self.requests.append(
            {
                "method": method,
                "path": path,
                "json": json,
                "params": params,
                "files": files,
            }
        )
        return self.response

    def get_status(self, task_id: str):
        if not self.statuses:
            raise AssertionError(f"unexpected status request for {task_id}")
        return self.statuses.pop(0)


class FakeDownloads(DownloadService):
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def save(
        self,
        task,
        directory,
        *,
        fallback_extension,
        fallback_task_id,
        cancel_check=None,
    ):
        self.calls.append({"task": task, "directory": directory})
        return (directory / f"{fallback_task_id}.mp4",)


def async_request(options: ExecutionOptions) -> OperationRequest:
    return OperationRequest(
        endpoint="/v1/video_generate",
        payload={"prompt": "test", "enabled": False, "seed": 0},
        execution=options,
        fallback_extension=".mp4",
    )


def test_async_operation_without_wait_returns_created_task(capsys):
    client = FakeClient({"task_id": "task-1", "status": "PENDING"})
    result = OperationService(client).execute(async_request(ExecutionOptions()))

    assert result.task == {"task_id": "task-1", "status": "PENDING"}
    assert serialize_operation_result(result) == {
        "task_id": "task-1",
        "status": "PENDING",
    }
    assert client.requests[0]["json"] == {"prompt": "test", "enabled": False, "seed": 0}
    assert capsys.readouterr().out == ""
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("response", [None, True, 42, "ok"])
def test_sync_scalar_response_is_returned_without_task_parsing(response):
    client = FakeClient(response)
    result = OperationService(client).execute(
        OperationRequest(endpoint="/v1/custom", payload={})
    )

    assert result.value == response
    assert result.task is None


def test_wait_and_download_use_one_execution_pipeline(make_temp_dir):
    target = make_temp_dir("wait-download")
    client = FakeClient(
        {"task_id": "task-2", "status": "PENDING"},
        statuses=[
            {"task_id": "task-2", "status": "SUCCESS", "result": "https://cdn.test/a"}
        ],
    )
    downloads = FakeDownloads()
    result = OperationService(client, download_service=downloads).execute(
        async_request(ExecutionOptions(download_directory=target, poll_interval=0))
    )

    assert result.task is not None and result.task["status"] == "SUCCESS"
    assert result.saved_files == (target / "task-2.mp4",)
    assert len(downloads.calls) == 1
    assert serialize_operation_result(result)["downloaded"] == [
        str(target / "task-2.mp4")
    ]


def test_failed_task_stops_before_downloading(make_temp_dir):
    target = make_temp_dir("failure")
    client = FakeClient(
        {"task_id": "task-3", "status": "PENDING"},
        statuses=[{"task_id": "task-3", "status": "FAILURE", "error": "Rejected"}],
    )
    downloads = FakeDownloads()

    with pytest.raises(TaskFailedError):
        OperationService(client, download_service=downloads).execute(
            async_request(ExecutionOptions(download_directory=target, poll_interval=0))
        )
    assert downloads.calls == []


def test_status_keeps_non_terminal_state_without_download():
    client = FakeClient({}, statuses=[{"task_id": "task-4", "status": "PROCESSING"}])
    result = OperationService(client).status("task-4", ExecutionOptions())

    assert serialize_operation_result(result) == {
        "task_id": "task-4",
        "status": "PROCESSING",
    }


def test_status_rejects_non_mapping_api_response():
    client = FakeClient({}, statuses=["not-a-task"])

    with pytest.raises(APIResponseError, match="invalid task status"):
        OperationService(client).status("task-4", ExecutionOptions())


def test_status_download_waits_for_success_before_saving(make_temp_dir):
    target = make_temp_dir("status-download")
    client = FakeClient(
        {},
        statuses=[
            {"task_id": "task-4", "status": "PROCESSING"},
            {
                "task_id": "task-4",
                "status": "SUCCESS",
                "result": "https://cdn.test/video",
            },
        ],
    )
    downloads = FakeDownloads()

    result = OperationService(client, download_service=downloads).status(
        "task-4",
        ExecutionOptions(download_directory=target, poll_interval=0),
        fallback_extension=".mp4",
    )

    assert result.task is not None and result.task["status"] == "SUCCESS"
    assert result.saved_files == (target / "task-4.mp4",)
    assert client.statuses == []
    assert len(downloads.calls) == 1


def test_polling_timeout_keeps_api_exit_code():
    client = FakeClient(
        {"task_id": "task-timeout", "status": "PENDING"},
        statuses=[{"task_id": "task-timeout", "status": "PROCESSING"}],
    )

    with pytest.raises(TaskPollingError) as error:
        OperationService(client).execute(
            async_request(ExecutionOptions(wait=True, timeout=0, poll_interval=0))
        )
    assert error.value.exit_code == 4


def test_video_edit_builder_validates_content_and_keeps_false(make_temp_dir):
    image = make_temp_dir("video-edit") / "reference.png"
    image.write_bytes(b"png")
    payload = build_video_edit_payload(
        prompt="edit",
        images=[str(image)],
        video=None,
        audio=None,
        duration=5,
        resolution="720p",
        aspect_ratio="16:9",
        generate_audio=False,
        callback_url=None,
    )

    assert payload["generate_audio"] is False
    assert payload["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    with pytest.raises(ValidationError):
        build_video_edit_payload(
            prompt="edit",
            images=[],
            video=None,
            audio=None,
            duration=5,
            resolution="720p",
            aspect_ratio="16:9",
            generate_audio=True,
            callback_url=None,
        )


def test_generic_video_edit_uses_service_and_encodes_local_media(make_temp_dir):
    image = make_temp_dir("generic-video-edit") / "reference.png"
    image.write_bytes(b"png")
    client = FakeClient({"task_id": "task-5", "status": "PENDING"})
    services = ApplicationServices.with_client(client)
    content = [
        {"type": "text", "text": "edit"},
        {
            "type": "image_url",
            "role": "reference_image",
            "image_url": {"url": str(image)},
        },
    ]
    schema = {"paths": {"/v1/video_edit": {"post": {"operationId": "video_edit"}}}}

    result = services.execute_generic(
        endpoint="video_edit",
        payload={
            "prompt": "edit",
            "model": "seedance2",
            "content": content,
            "enabled": False,
            "seed": 0,
        },
        method=None,
        execution=ExecutionOptions(),
        schema_loader=lambda _: (schema, "test"),
    )

    assert result.task is not None
    sent = client.requests[0]["json"]
    assert sent["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert sent["enabled"] is False and sent["seed"] == 0


def test_generic_wan_video_edit_encodes_flat_local_media(make_temp_dir):
    target = make_temp_dir("generic-wan-video-edit")
    video = target / "source.mp4"
    image = target / "reference.png"
    video.write_bytes(b"video")
    image.write_bytes(b"image")
    client = FakeClient({"task_id": "task-wan", "status": "PENDING"})
    services = ApplicationServices.with_client(client)
    schema = {"paths": {"/v1/video_edit": {"post": {"operationId": "video_edit"}}}}

    services.execute_generic(
        endpoint="video_edit",
        payload={
            "prompt": "change outfit",
            "model": "wan2.7",
            "video_url": str(video),
            "reference_image_urls": [str(image)],
        },
        method=None,
        execution=ExecutionOptions(),
        schema_loader=lambda _: (schema, "test"),
    )

    sent = client.requests[0]["json"]
    assert sent["video_url"].startswith("data:video/mp4;base64,")
    assert sent["reference_image_urls"][0].startswith("data:image/png;base64,")


def test_generic_aleph_video_edit_encodes_local_keyframe(make_temp_dir):
    target = make_temp_dir("generic-aleph-video-edit")
    video = target / "source.mp4"
    image = target / "keyframe.png"
    video.write_bytes(b"video")
    image.write_bytes(b"image")
    client = FakeClient({"task_id": "task-aleph", "status": "PENDING"})
    services = ApplicationServices.with_client(client)
    schema = {"paths": {"/v1/video_edit": {"post": {"operationId": "video_edit"}}}}

    services.execute_generic(
        endpoint="video_edit",
        payload={
            "model": "aleph2",
            "video_url": str(video),
            "keyframes": [{"image_url": str(image), "seconds": 1}],
        },
        method=None,
        execution=ExecutionOptions(),
        schema_loader=lambda _: (schema, "test"),
    )

    sent = client.requests[0]["json"]
    assert sent["video_url"].startswith("data:video/mp4;base64,")
    assert sent["keyframes"][0]["image_url"].startswith("data:image/png;base64,")


def test_asr_download_service_preserves_text_and_structured_json(make_temp_dir):
    target = make_temp_dir("asr")
    service = DownloadService()
    text_paths = service.save(
        {"task_id": "asr-text", "result": "hello"},
        target,
        fallback_extension=".txt",
        fallback_task_id="asr-text",
    )
    json_paths = service.save(
        {
            "task_id": "asr-json",
            "result": {"text": "hello", "segments": [{"start": 0}]},
        },
        target,
        fallback_extension=".txt",
        fallback_task_id="asr-json",
    )

    assert text_paths[0].read_text(encoding="utf-8") == "hello"
    assert json.loads(json_paths[0].read_text(encoding="utf-8"))["segments"] == [
        {"start": 0}
    ]


def test_download_errors_keep_file_exit_code(make_temp_dir):
    target = make_temp_dir("download-error")

    class BrokenDownloads(DownloadService):
        def save(self, *args, **kwargs):
            raise FileWriteError("Unable to write result file")

    client = FakeClient(
        {"task_id": "task-6", "status": "PENDING"},
        statuses=[{"task_id": "task-6", "status": "SUCCESS", "result": "x"}],
    )
    with pytest.raises(FileWriteError) as error:
        OperationService(client, download_service=BrokenDownloads()).execute(
            async_request(ExecutionOptions(download_directory=target, poll_interval=0))
        )
    assert error.value.exit_code == 5
