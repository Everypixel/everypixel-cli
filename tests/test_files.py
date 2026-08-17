import base64
from pathlib import Path

import httpx
import pytest
import respx

from everypixel_cli.files import (
    ResultDownload,
    download_urls,
    file_to_data_uri,
    save_inline_result,
)


FIXTURES = Path(__file__).parent / "fixtures"


def test_file_to_data_uri():
    image = FIXTURES / "sample.png"

    assert (
        file_to_data_uri(image)
        == f"data:image/png;base64,{base64.b64encode(image.read_bytes()).decode('ascii')}"
    )


@respx.mock
def test_download_urls_uses_video_audio_and_text_extensions(make_temp_dir):
    output_dir = make_temp_dir("extensions")
    respx.get("https://cdn.test/video").mock(
        return_value=httpx.Response(
            200, content=b"video", headers={"content-type": "video/mp4"}
        )
    )
    respx.get("https://cdn.test/audio").mock(
        return_value=httpx.Response(
            200, content=b"audio", headers={"content-type": "audio/mpeg"}
        )
    )
    respx.get("https://cdn.test/transcript").mock(
        return_value=httpx.Response(
            200, content=b"text", headers={"content-type": "text/plain"}
        )
    )

    paths = download_urls(
        [
            "https://cdn.test/video",
            "https://cdn.test/audio",
            "https://cdn.test/transcript",
        ],
        output_dir,
        fallback_ext=".bin",
    )

    assert paths == [
        str(output_dir / "result-1.mp4"),
        str(output_dir / "result-2.mp3"),
        str(output_dir / "result-3.txt"),
    ]
    assert (output_dir / "result-1.mp4").read_bytes() == b"video"
    assert (output_dir / "result-2.mp3").read_bytes() == b"audio"
    assert (output_dir / "result-3.txt").read_bytes() == b"text"


@respx.mock
def test_download_urls_uses_task_id_and_mime_extension(make_temp_dir):
    output_dir = make_temp_dir("mime-extension")
    respx.get("https://cdn.test/image").mock(
        return_value=httpx.Response(
            200, content=b"image", headers={"content-type": "image/png"}
        )
    )

    paths = download_urls(
        ["https://cdn.test/image"], output_dir, fallback_ext=".jpg", task_id="abc"
    )

    assert paths == [str(output_dir / "abc.png")]
    assert (output_dir / "abc.png").read_bytes() == b"image"


@respx.mock
def test_download_urls_uses_safe_fallback_extension(make_temp_dir):
    output_dir = make_temp_dir("fallback-extension")
    respx.get("https://cdn.test/result").mock(
        return_value=httpx.Response(200, content=b"data")
    )

    paths = download_urls(
        [ResultDownload("https://cdn.test/result", ".mp4")],
        output_dir,
        task_id="../abc",
    )

    assert paths == [str(output_dir / "abc.mp4")]
    assert (output_dir / "abc.mp4").read_bytes() == b"data"


@respx.mock
def test_cancelled_download_does_not_overwrite_existing_result(make_temp_dir):
    output_dir = make_temp_dir("cancelled-download")
    target = output_dir / "task.png"
    target.write_bytes(b"existing")
    cancelled = False

    def cancel_after_response(_request: httpx.Request) -> httpx.Response:
        nonlocal cancelled
        cancelled = True
        return httpx.Response(
            200,
            content=b"replacement",
            headers={"content-type": "image/png"},
        )

    respx.get("https://cdn.test/image").mock(side_effect=cancel_after_response)

    def cancel_during_download() -> None:
        if cancelled:
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        download_urls(
            ["https://cdn.test/image"],
            output_dir,
            task_id="task",
            cancel_check=cancel_during_download,
        )

    assert target.read_bytes() == b"existing"
    assert not (output_dir / ".task.png.part").exists()


@respx.mock
def test_cancelled_chunked_download_removes_partial_file(make_temp_dir):
    class ChunkedStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b"first"
            yield b"second"

    output_dir = make_temp_dir("cancelled-chunked-download")
    target = output_dir / "task.bin"
    target.write_bytes(b"existing")
    respx.get("https://cdn.test/chunked").mock(
        return_value=httpx.Response(200, stream=ChunkedStream())
    )
    checks = 0

    def cancel_after_first_chunk() -> None:
        nonlocal checks
        checks += 1
        if checks == 3:
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        download_urls(
            ["https://cdn.test/chunked"],
            output_dir,
            task_id="task",
            cancel_check=cancel_after_first_chunk,
        )

    assert target.read_bytes() == b"existing"
    assert not (output_dir / ".task.bin.part").exists()


def test_save_inline_asr_transcript_as_task_txt(make_temp_dir):
    output_dir = make_temp_dir("asr-txt")
    paths = save_inline_result(
        "hello world", output_dir, task_id="asr-task", fallback_ext=".txt"
    )

    assert paths == [str(output_dir / "asr-task.txt")]
    assert (output_dir / "asr-task.txt").read_text(encoding="utf-8") == "hello world"


def test_save_inline_structured_asr_json_keeps_fields(make_temp_dir):
    output_dir = make_temp_dir("asr-json")
    result = {
        "text": "hello",
        "language": "en",
        "segments": [{"start": 0, "end": 1, "text": "hello"}],
    }

    paths = save_inline_result(
        result, output_dir, task_id="asr-task", fallback_ext=".txt"
    )

    assert paths == [str(output_dir / "asr-task.json")]
    assert '"language": "en"' in (output_dir / "asr-task.json").read_text(
        encoding="utf-8"
    )
    assert '"segments"' in (output_dir / "asr-task.json").read_text(encoding="utf-8")
