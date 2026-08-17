"""Everypixel CLI Typer application.

This module defines the command tree and runtime option resolution.
HTTP logic lives in client.py; Pydantic payload validation lives in schemas.py.
"""

from __future__ import annotations

import sys
import traceback
import webbrowser
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, NoReturn, Optional

import typer
from click import BadParameter, ClickException
from typer.core import TyperGroup

try:  # Typer 0.26+ vendors Click while older supported releases import it.
    from typer._click.exceptions import ClickException as TyperClickException
except ImportError:  # pragma: no cover - exercised with older Typer versions.
    TyperClickException = ClickException  # type: ignore[misc, assignment]

from .application import ApplicationServices, ExecutionOptions
from .application.serialization import serialize_operation_result
from .application.services import parse_generic_payload
from .client import APIClient
from .config import (
    DEFAULT_BASE_URL,
    ProfileConfig,
    delete_credentials,
    load_config,
    resolved_settings,
    safe_config_payload,
    save_config,
    save_credentials,
    set_config_value,
    use_profile,
)
from .errors import (
    CLIError,
    InternalCLIError,
    ValidationCLIError,
    mask_secret,
    normalize_exception,
)
from .openapi import load_schema
from .output import emit_error, emit_human, emit_json


def json_output_requested(args: Sequence[str]) -> bool:
    """Detect JSON mode before Click has parsed the full command line."""

    for index, argument in enumerate(args):
        if argument == "--":
            break
        if argument in {"--output-json", "-j", "--jq"}:
            return True
        if argument.startswith("--jq="):
            return True
        if (
            argument == "--output"
            and index + 1 < len(args)
            and args[index + 1] == "json"
        ):
            return True
        if argument == "--output=json":
            return True
    return False


class JSONErrorBoundaryGroup(TyperGroup):
    """Render Click parsing failures through the shared JSON error contract."""

    def main(
        self,
        args: Sequence[str] | None = None,
        prog_name: str | None = None,
        complete_var: str | None = None,
        standalone_mode: bool = True,
        windows_expand_args: bool = True,
        **extra: Any,
    ) -> Any:
        raw_args = list(args) if args is not None else sys.argv[1:]
        if not json_output_requested(raw_args):
            return super().main(
                args=args,
                prog_name=prog_name,
                complete_var=complete_var,
                standalone_mode=standalone_mode,
                windows_expand_args=windows_expand_args,
                **extra,
            )

        try:
            result = super().main(
                args=args,
                prog_name=prog_name,
                complete_var=complete_var,
                standalone_mode=False,
                windows_expand_args=windows_expand_args,
                **extra,
            )
        except (ClickException, TyperClickException) as exc:
            emit_error(
                ValidationCLIError(exc.format_message()),
                output_json=True,
            )
            if not standalone_mode:
                raise
            raise SystemExit(exc.exit_code) from exc

        if standalone_mode:
            raise SystemExit(result if isinstance(result, int) else 0)
        return result


app = typer.Typer(
    cls=JSONErrorBoundaryGroup,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
auth_app = typer.Typer(no_args_is_help=True)
config_app = typer.Typer(no_args_is_help=True)
profile_app = typer.Typer(no_args_is_help=True)
image_app = typer.Typer(no_args_is_help=True)
video_app = typer.Typer(no_args_is_help=True)
lipsync_app = typer.Typer(no_args_is_help=True)
audio_app = typer.Typer(no_args_is_help=True)
docs_app = typer.Typer(no_args_is_help=True)
schema_app = typer.Typer(no_args_is_help=True)
mcp_app = typer.Typer(no_args_is_help=True)

app.add_typer(auth_app, name="auth")
app.add_typer(config_app, name="config")
config_app.add_typer(profile_app, name="profile")
app.add_typer(image_app, name="image")
app.add_typer(video_app, name="video")
app.add_typer(lipsync_app, name="lipsync")
app.add_typer(audio_app, name="audio")
app.add_typer(docs_app, name="docs")
app.add_typer(schema_app, name="schema")
app.add_typer(mcp_app, name="mcp")


class OutputMode(str, Enum):
    human = "human"
    json = "json"


def positive_float(value: float) -> float:
    """Reject zero and negative wait controls at the CLI boundary."""

    if value <= 0:
        raise typer.BadParameter("must be greater than 0")
    return value


WaitOption = Annotated[
    Optional[bool], typer.Option("--wait/--no-wait", help="Override wait mode.")
]
DownloadOption = Annotated[
    Optional[Path],
    typer.Option("--download", "-d", help="Download result URLs into a directory."),
]
OutputJsonOption = Annotated[
    bool, typer.Option("--output-json", "-j", help="Alias for --output json.")
]
JqOption = Annotated[
    Optional[str], typer.Option("--jq", help="Apply jq expression to JSON output.")
]


@dataclass
class Runtime:
    """Resolved settings for one CLI invocation."""

    base_url: str
    profile: str
    client_id: str | None
    client_secret: str | None
    output_json: bool
    jq: str | None
    wait: bool
    timeout: float
    poll_interval: float
    download: Path | None
    no_color: bool
    debug: bool
    _services: ApplicationServices | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def client(self) -> APIClient:
        """Create an API client with the current credentials."""

        return APIClient(
            base_url=self.base_url,
            client_id=self.client_id,
            client_secret=self.client_secret,
        )

    def services(self) -> ApplicationServices:
        """Create the CLI-independent operation services for this invocation."""

        if self._services is None:
            self._services = ApplicationServices.with_client(self.client())
        return self._services

    def close(self) -> None:
        """Close transport resources created for this invocation."""

        if self._services is not None:
            self._services.close()

    def execution_options(self, *, wait: bool | None = None) -> ExecutionOptions:
        """Build application execution options from the resolved CLI runtime."""

        return ExecutionOptions(
            wait=self.wait if wait is None else wait,
            download_directory=self.download,
            timeout=self.timeout,
            poll_interval=self.poll_interval,
        )


def get_runtime(ctx: typer.Context) -> Runtime:
    """Read Runtime from Typer context or fail with a CLI error."""

    if not isinstance(ctx.obj, Runtime):
        raise CLIError("Runtime context is not initialized")
    return ctx.obj


@app.callback()
def main(
    ctx: typer.Context,
    base_url: Annotated[
        Optional[str],
        typer.Option("--base-url", help="Override API base URL for this run."),
    ] = None,
    profile: Annotated[
        Optional[str], typer.Option("--profile", help="Config profile name.")
    ] = None,
    dev: Annotated[bool, typer.Option("--dev", help="Use the dev profile.")] = False,
    output: Annotated[
        OutputMode, typer.Option("--output", help="Output mode: human or json.")
    ] = OutputMode.human,
    output_json: Annotated[
        bool, typer.Option("--output-json", "-j", help="Alias for --output json.")
    ] = False,
    jq: Annotated[
        Optional[str], typer.Option("--jq", help="Apply jq expression to JSON output.")
    ] = None,
    wait: Annotated[
        bool, typer.Option("--wait/--no-wait", help="Wait for async task completion.")
    ] = True,
    timeout: Annotated[
        float,
        typer.Option(
            "--timeout",
            callback=positive_float,
            help="Task wait timeout in seconds.",
        ),
    ] = 300.0,
    poll_interval: Annotated[
        float,
        typer.Option(
            "--poll-interval",
            callback=positive_float,
            help="Task polling interval in seconds.",
        ),
    ] = 2.0,
    download: Annotated[
        Optional[Path],
        typer.Option("--download", "-d", help="Download result URLs into a directory."),
    ] = None,
    no_color: Annotated[
        bool, typer.Option("--no-color", help="Disable colored human output.")
    ] = False,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug",
            help="Include sanitized stack frames for unexpected internal errors.",
        ),
    ] = False,
) -> None:
    """Global callback that resolves settings shared by all commands."""

    try:
        settings = resolved_settings(profile=profile, dev=dev, base_url=base_url)
    except Exception as exc:
        error = error_with_debug_context(normalized_error(exc), exc, enabled=debug)
        emit_error(
            error,
            output_json=output_json or output == OutputMode.json or jq is not None,
            no_color=no_color,
        )
        raise typer.Exit(error.exit_code) from exc
    runtime = Runtime(
        base_url=settings["base_url"],
        profile=settings["profile"],
        client_id=settings["client_id"],
        client_secret=settings["client_secret"],
        output_json=output_json or output == OutputMode.json or jq is not None,
        jq=jq,
        wait=wait,
        timeout=timeout,
        poll_interval=poll_interval,
        download=download,
        no_color=no_color,
        debug=debug,
    )
    ctx.obj = runtime
    ctx.call_on_close(runtime.close)


def finish(ctx: typer.Context, payload: Any, *, title: str | None = None) -> None:
    """Print payload in the selected output format."""

    runtime = get_runtime(ctx)
    if hasattr(payload, "saved_files") and hasattr(payload, "value"):
        payload = serialize_operation_result(payload)
    if runtime.output_json:
        emit_json(payload, runtime.jq)
    else:
        emit_human(payload, title=title, no_color=runtime.no_color)


def normalized_error(exc: Exception) -> CLIError:
    """Convert expected technical exceptions to application errors."""

    if isinstance(exc, BadParameter):
        return ValidationCLIError(str(exc))
    return normalize_exception(exc)


def error_with_debug_context(
    error: CLIError,
    exc: Exception,
    *,
    enabled: bool,
) -> CLIError:
    """Attach stack locations without exception messages, locals, or payloads."""

    if not enabled or not isinstance(error, InternalCLIError):
        return error
    frames = traceback.extract_tb(exc.__traceback__)[-20:]
    return InternalCLIError(
        error.message,
        details={
            "exception_type": type(exc).__name__,
            "stack": [
                {
                    "file": sanitized_stack_filename(frame.filename),
                    "line": frame.lineno,
                    "function": frame.name,
                }
                for frame in frames
            ],
        },
    )


def sanitized_stack_filename(filename: str) -> str:
    """Keep project-relative locations while hiding host-specific path prefixes."""

    path = Path(filename)
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except (OSError, ValueError):
        return path.name


def stop_with_error(ctx: typer.Context, exc: Exception) -> NoReturn:
    """Render one error through the sole CLI output boundary and exit."""

    runtime = ctx.obj if isinstance(ctx.obj, Runtime) else None
    error = error_with_debug_context(
        normalized_error(exc),
        exc,
        enabled=bool(runtime and runtime.debug),
    )
    emit_error(
        error,
        output_json=bool(runtime and runtime.output_json),
        no_color=bool(runtime and runtime.no_color),
    )
    raise typer.Exit(error.exit_code) from exc


def run_action(ctx: typer.Context, action: Callable[[], Any]) -> None:
    """Run a command action and handle all expected application errors."""

    try:
        finish(ctx, action())
    except typer.Exit:
        raise
    except (
        Exception
    ) as exc:  # Boundary deliberately turns unexpected errors into clean CLI failures.
        stop_with_error(ctx, exc)


def apply_common_options(
    ctx: typer.Context,
    *,
    wait_option: bool | None = None,
    download: Path | None = None,
    output_json: bool = False,
    jq_expr: str | None = None,
) -> Runtime:
    """Apply command-local common flags on top of global runtime settings."""

    runtime = get_runtime(ctx)
    if output_json or jq_expr is not None:
        runtime.output_json = True
    if jq_expr is not None:
        runtime.jq = jq_expr
    if download is not None:
        runtime.download = download
    if wait_option is not None:
        runtime.wait = wait_option
    return runtime


@auth_app.command("configure")
def auth_configure(
    ctx: typer.Context,
    client_id: Annotated[str, typer.Option("--client-id", prompt=True)],
    client_secret: Annotated[
        str, typer.Option("--client-secret", prompt=True, hide_input=True)
    ],
    base_url: Annotated[
        Optional[str],
        typer.Option("--base-url", help="Persist base URL for the current profile."),
    ] = None,
) -> None:
    """Save Basic Auth credentials for the current profile."""

    runtime = get_runtime(ctx)

    def action() -> dict[str, Any]:
        keyring_ok = save_credentials(runtime.profile, client_id, client_secret)
        if base_url:
            set_config_value(runtime.profile, "base_url", base_url)
        return {
            "profile": runtime.profile,
            "client_id": client_id,
            "base_url": base_url or runtime.base_url,
            "storage": "keyring" if keyring_ok else "config_file",
        }

    run_action(ctx, action)


@auth_app.command("check")
def auth_check(ctx: typer.Context) -> None:
    """Check that stored credentials are accepted by the API."""

    runtime = get_runtime(ctx)

    def action() -> dict[str, Any]:
        payload = serialize_operation_result(runtime.services().check_auth())
        return {
            **payload,
            "profile": runtime.profile,
            "base_url": runtime.base_url,
        }

    run_action(ctx, action)


@auth_app.command("whoami")
def auth_whoami(ctx: typer.Context) -> None:
    """Show the active profile and masked credentials."""

    runtime = get_runtime(ctx)
    run_action(
        ctx,
        lambda: {
            "profile": runtime.profile,
            "base_url": runtime.base_url,
            "client_id": runtime.client_id,
            "client_secret": mask_secret(runtime.client_secret),
        },
    )


@auth_app.command("logout")
def auth_logout(ctx: typer.Context) -> None:
    """Remove credentials from the current profile."""

    runtime = get_runtime(ctx)

    def action() -> dict[str, Any]:
        delete_credentials(runtime.profile)
        return {"status": "ok", "profile": runtime.profile}

    run_action(ctx, action)


@config_app.command("set")
def config_set(ctx: typer.Context, key: str, value: str) -> None:
    """Set a supported option for the current profile."""

    runtime = get_runtime(ctx)

    def action() -> dict[str, Any]:
        set_config_value(runtime.profile, key, value)
        return {"status": "ok", "profile": runtime.profile, key: value}

    run_action(ctx, action)


@config_app.command("get")
def config_get(ctx: typer.Context, key: str) -> None:
    """Show a setting from the current profile."""

    runtime = get_runtime(ctx)

    def action() -> dict[str, Any]:
        if key != "base_url":
            raise ValidationCLIError("Only base_url is supported")
        return {"profile": runtime.profile, key: runtime.base_url}

    run_action(ctx, action)


@config_app.command("list")
def config_list(ctx: typer.Context) -> None:
    """Show local configuration without exposing secrets."""

    run_action(ctx, lambda: safe_config_payload(load_config()))


@profile_app.command("create")
def profile_create(
    ctx: typer.Context,
    name: str,
    base_url: Annotated[str, typer.Option("--base-url")] = DEFAULT_BASE_URL,
) -> None:
    run_action(ctx, lambda: create_profile_payload(name, base_url))


def create_profile_payload(name: str, base_url: str) -> dict[str, Any]:
    """Create or update a profile and return a result payload."""

    config = load_config()
    config.profiles[name] = config.profiles.get(name) or ProfileConfig()
    config.profiles[name].base_url = base_url
    save_config(config)
    return {"status": "ok", "profile": name, "base_url": base_url}


@profile_app.command("use")
def profile_use(ctx: typer.Context, name: str) -> None:
    """Make a profile active for future runs."""

    def action() -> dict[str, Any]:
        use_profile(name)
        return {"status": "ok", "profile": name}

    run_action(ctx, action)


@image_app.command("generate")
def image_generate(
    ctx: typer.Context,
    prompt: str,
    model: Annotated[str, typer.Option("--model")] = "zimage",
    size: Annotated[str, typer.Option("--size")] = "square",
    style: Annotated[Optional[str], typer.Option("--style")] = None,
    image: Annotated[Optional[str], typer.Option("--image")] = None,
    resolution: Annotated[Optional[str], typer.Option("--resolution")] = None,
    seed: Annotated[int, typer.Option("--seed")] = -1,
    callback_url: Annotated[Optional[str], typer.Option("--callback-url")] = None,
    wait_option: WaitOption = None,
    download: DownloadOption = None,
    output_json: OutputJsonOption = False,
    jq_expr: JqOption = None,
) -> None:
    """Start image generation."""

    runtime = apply_common_options(
        ctx,
        wait_option=wait_option,
        download=download,
        output_json=output_json,
        jq_expr=jq_expr,
    )
    run_action(
        ctx,
        lambda: runtime.services().execute_image_generate(
            prompt=prompt,
            model=model,
            image_size=size,
            style=style,
            image=image,
            resolution=resolution,
            seed=seed,
            callback_url=callback_url,
            execution=runtime.execution_options(),
        ),
    )


@image_app.command("edit")
def image_edit(
    ctx: typer.Context,
    prompt: str,
    image: Annotated[
        list[str],
        typer.Option("--image", help="Image URL or local path. Can be repeated."),
    ],
    model: Annotated[str, typer.Option("--model")] = "flux2",
    size: Annotated[Optional[str], typer.Option("--size")] = None,
    resolution: Annotated[Optional[str], typer.Option("--resolution")] = None,
    seed: Annotated[int, typer.Option("--seed")] = -1,
    callback_url: Annotated[Optional[str], typer.Option("--callback-url")] = None,
    wait_option: WaitOption = None,
    download: DownloadOption = None,
    output_json: OutputJsonOption = False,
    jq_expr: JqOption = None,
) -> None:
    """Start image editing."""

    runtime = apply_common_options(
        ctx,
        wait_option=wait_option,
        download=download,
        output_json=output_json,
        jq_expr=jq_expr,
    )
    run_action(
        ctx,
        lambda: runtime.services().execute_image_edit(
            prompt=prompt,
            images=image,
            model=model,
            image_size=size,
            resolution=resolution,
            seed=seed,
            callback_url=callback_url,
            execution=runtime.execution_options(),
        ),
    )


@image_app.command("upscale")
def image_upscale(
    ctx: typer.Context,
    image: Annotated[Optional[str], typer.Option("--image")] = None,
    task_id: Annotated[Optional[str], typer.Option("--task-id")] = None,
    model: Annotated[str, typer.Option("--model")] = "seedvr2",
    callback_url: Annotated[Optional[str], typer.Option("--callback-url")] = None,
    wait_option: WaitOption = None,
    download: DownloadOption = None,
    output_json: OutputJsonOption = False,
    jq_expr: JqOption = None,
) -> None:
    """Start image upscale from a URL/file or task_id."""

    runtime = apply_common_options(
        ctx,
        wait_option=wait_option,
        download=download,
        output_json=output_json,
        jq_expr=jq_expr,
    )
    run_action(
        ctx,
        lambda: runtime.services().execute_image_upscale(
            image=image,
            task_id=task_id,
            model=model,
            callback_url=callback_url,
            execution=runtime.execution_options(),
        ),
    )


@image_app.command("angles")
def image_angles(
    ctx: typer.Context,
    image: Annotated[str, typer.Option("--image")],
    azimuth: str = "front",
    elevation: str = "eye_level",
    distance: str = "medium",
    prompt: Optional[str] = None,
    wait_option: WaitOption = None,
    download: DownloadOption = None,
    output_json: OutputJsonOption = False,
    jq_expr: JqOption = None,
) -> None:
    """Change image angle through a dedicated endpoint."""

    runtime = apply_common_options(
        ctx,
        wait_option=wait_option,
        download=download,
        output_json=output_json,
        jq_expr=jq_expr,
    )
    run_action(
        ctx,
        lambda: runtime.services().execute_image_angles(
            image=image,
            azimuth=azimuth,
            elevation=elevation,
            distance=distance,
            prompt=prompt,
            execution=runtime.execution_options(),
        ),
    )


@image_app.command("colors")
def image_colors(
    ctx: typer.Context,
    image: Annotated[str, typer.Option("--image")],
    reference: Annotated[str, typer.Option("--reference")],
    wait_option: WaitOption = None,
    download: DownloadOption = None,
    output_json: OutputJsonOption = False,
    jq_expr: JqOption = None,
) -> None:
    """Transfer reference colors to the source image."""

    runtime = apply_common_options(
        ctx,
        wait_option=wait_option,
        download=download,
        output_json=output_json,
        jq_expr=jq_expr,
    )
    run_action(
        ctx,
        lambda: runtime.services().execute_image_colors(
            image=image,
            reference=reference,
            execution=runtime.execution_options(),
        ),
    )


@video_app.command("generate")
def video_generate(
    ctx: typer.Context,
    prompt: str,
    model: str = "ltx23",
    duration: Optional[int] = None,
    resolution: Optional[str] = None,
    aspect_ratio: Annotated[str, typer.Option("--aspect-ratio")] = "16:9",
    lora_high_url: Annotated[Optional[str], typer.Option("--lora-high-url")] = None,
    lora_low_url: Annotated[Optional[str], typer.Option("--lora-low-url")] = None,
    reference_image: Annotated[
        list[str],
        typer.Option(
            "--reference-image",
            help="Wan or Veo reference image URL/local path. Can be repeated.",
        ),
    ] = [],
    reference_video: Annotated[
        list[str],
        typer.Option(
            "--reference-video",
            help="Wan reference video URL or local path. Can be repeated.",
        ),
    ] = [],
    seed: Optional[int] = None,
    generate_audio: Annotated[
        Optional[bool], typer.Option("--generate-audio/--no-generate-audio")
    ] = None,
    callback_url: Annotated[Optional[str], typer.Option("--callback-url")] = None,
    wait_option: WaitOption = None,
    download: DownloadOption = None,
    output_json: OutputJsonOption = False,
    jq_expr: JqOption = None,
) -> None:
    """Start text-to-video generation."""

    runtime = apply_common_options(
        ctx,
        wait_option=wait_option,
        download=download,
        output_json=output_json,
        jq_expr=jq_expr,
    )
    run_action(
        ctx,
        lambda: runtime.services().execute_video_generate(
            prompt=prompt,
            model=model,
            duration=duration,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            lora_high_url=lora_high_url,
            lora_low_url=lora_low_url,
            reference_images=reference_image,
            reference_videos=reference_video,
            seed=seed,
            generate_audio=generate_audio,
            callback_url=callback_url,
            execution=runtime.execution_options(),
        ),
    )


@video_app.command("edit")
def video_edit(
    ctx: typer.Context,
    prompt: Annotated[
        Optional[str], typer.Argument(help="Edit prompt; optional for Aleph 2.")
    ] = None,
    model: Annotated[str, typer.Option("--model")] = "seedance2",
    image: Annotated[
        list[str], typer.Option("--image", help="Reference image URL or local path.")
    ] = [],
    video: Annotated[
        Optional[str],
        typer.Option("--video", help="Reference video URL or local path."),
    ] = None,
    audio: Annotated[
        Optional[str],
        typer.Option("--audio", help="Reference audio URL or local path."),
    ] = None,
    duration: Optional[int] = None,
    resolution: Optional[str] = None,
    aspect_ratio: Annotated[Optional[str], typer.Option("--aspect-ratio")] = None,
    seed: Annotated[Optional[int], typer.Option("--seed")] = None,
    keyframe: Annotated[
        list[str],
        typer.Option(
            "--keyframe",
            help=(
                "Aleph 2 keyframe JSON with image_url and seconds or at. "
                "Can be repeated."
            ),
        ),
    ] = [],
    public_figure_threshold: Annotated[
        Optional[str], typer.Option("--public-figure-threshold")
    ] = None,
    generate_audio: Annotated[
        Optional[bool], typer.Option("--generate-audio/--no-generate-audio")
    ] = None,
    callback_url: Annotated[Optional[str], typer.Option("--callback-url")] = None,
    wait_option: WaitOption = None,
    download: DownloadOption = None,
    output_json: OutputJsonOption = False,
    jq_expr: JqOption = None,
) -> None:
    """Edit video with Seedance, Kling, Wan, or Aleph media."""

    runtime = apply_common_options(
        ctx,
        wait_option=wait_option,
        download=download,
        output_json=output_json,
        jq_expr=jq_expr,
    )
    run_action(
        ctx,
        lambda: runtime.services().execute_video_edit(
            prompt=prompt,
            model=model,
            images=image,
            video=video,
            audio=audio,
            duration=duration,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            seed=seed,
            keyframes=keyframe,
            public_figure_threshold=public_figure_threshold,
            generate_audio=generate_audio,
            callback_url=callback_url,
            execution=runtime.execution_options(),
        ),
    )


@video_app.command("from-image")
def video_from_image(
    ctx: typer.Context,
    prompt: str,
    image: Annotated[str, typer.Option("--image")],
    model: str = "ltx23",
    duration: Optional[int] = None,
    resolution: Optional[str] = None,
    aspect_ratio: Annotated[str, typer.Option("--aspect-ratio")] = "16:9",
    seed: Optional[int] = None,
    callback_url: Annotated[Optional[str], typer.Option("--callback-url")] = None,
    wait_option: WaitOption = None,
    download: DownloadOption = None,
    output_json: OutputJsonOption = False,
    jq_expr: JqOption = None,
) -> None:
    """Start image-to-video generation."""

    runtime = apply_common_options(
        ctx,
        wait_option=wait_option,
        download=download,
        output_json=output_json,
        jq_expr=jq_expr,
    )
    run_action(
        ctx,
        lambda: runtime.services().execute_video_generate(
            prompt=prompt,
            model=model,
            duration=duration,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            image=image,
            seed=seed,
            callback_url=callback_url,
            execution=runtime.execution_options(),
        ),
    )


@video_app.command("first-last")
def video_first_last(
    ctx: typer.Context,
    prompt: str,
    image: Annotated[str, typer.Option("--image")],
    last_image: Annotated[str, typer.Option("--last-image")],
    model: str = "ltx23",
    duration: Optional[int] = None,
    resolution: Optional[str] = None,
    aspect_ratio: Annotated[str, typer.Option("--aspect-ratio")] = "16:9",
    seed: Optional[int] = None,
    callback_url: Annotated[Optional[str], typer.Option("--callback-url")] = None,
    wait_option: WaitOption = None,
    download: DownloadOption = None,
    output_json: OutputJsonOption = False,
    jq_expr: JqOption = None,
) -> None:
    """Generate video between first and last frames."""

    runtime = apply_common_options(
        ctx,
        wait_option=wait_option,
        download=download,
        output_json=output_json,
        jq_expr=jq_expr,
    )
    run_action(
        ctx,
        lambda: runtime.services().execute_video_generate(
            prompt=prompt,
            model=model,
            duration=duration,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            image=image,
            last_image=last_image,
            seed=seed,
            callback_url=callback_url,
            execution=runtime.execution_options(),
        ),
    )


@video_app.command("upscale")
def video_upscale(
    ctx: typer.Context,
    video: Annotated[Optional[str], typer.Option("--video")] = None,
    task_id: Annotated[Optional[str], typer.Option("--task-id")] = None,
    resolution: Annotated[
        str,
        typer.Option(
            "--resolution",
            help="Output resolution: 720p, 1080p, or 1440p (up to 20s at 1440p).",
        ),
    ] = "1080p",
    wait_option: WaitOption = None,
    download: DownloadOption = None,
    output_json: OutputJsonOption = False,
    jq_expr: JqOption = None,
) -> None:
    """Start video upscale from a URL/file or task_id.

    Source videos longer than 20 seconds cannot be upscaled to 1440p.
    """

    runtime = apply_common_options(
        ctx,
        wait_option=wait_option,
        download=download,
        output_json=output_json,
        jq_expr=jq_expr,
    )
    run_action(
        ctx,
        lambda: runtime.services().execute_video_upscale(
            video=video,
            task_id=task_id,
            resolution=resolution,
            execution=runtime.execution_options(),
        ),
    )


@lipsync_app.command("video")
def lipsync_video(
    ctx: typer.Context,
    video: Annotated[str, typer.Option("--video")],
    audio: Annotated[str, typer.Option("--audio")],
    resolution: str = "480p",
    prompt: Optional[str] = None,
    seed: int = -1,
    callback_url: Annotated[Optional[str], typer.Option("--callback-url")] = None,
    wait_option: WaitOption = None,
    download: DownloadOption = None,
    output_json: OutputJsonOption = False,
    jq_expr: JqOption = None,
) -> None:
    """Synchronize lips in a video using an audio track."""

    runtime = apply_common_options(
        ctx,
        wait_option=wait_option,
        download=download,
        output_json=output_json,
        jq_expr=jq_expr,
    )
    run_action(
        ctx,
        lambda: runtime.services().execute_lipsync_video(
            video=video,
            audio=audio,
            resolution=resolution,
            prompt=prompt,
            seed=seed,
            callback_url=callback_url,
            execution=runtime.execution_options(),
        ),
    )


@lipsync_app.command("image")
def lipsync_image(
    ctx: typer.Context,
    image: Annotated[str, typer.Option("--image")],
    audio: Annotated[str, typer.Option("--audio")],
    model: str = "inftalk",
    resolution: str = "480p",
    prompt: Optional[str] = None,
    seed: int = -1,
    callback_url: Annotated[Optional[str], typer.Option("--callback-url")] = None,
    wait_option: WaitOption = None,
    download: DownloadOption = None,
    output_json: OutputJsonOption = False,
    jq_expr: JqOption = None,
) -> None:
    """Create a lipsync video from a still image and audio."""

    runtime = apply_common_options(
        ctx,
        wait_option=wait_option,
        download=download,
        output_json=output_json,
        jq_expr=jq_expr,
    )
    run_action(
        ctx,
        lambda: runtime.services().execute_lipsync_image(
            image=image,
            audio=audio,
            model=model,
            resolution=resolution,
            prompt=prompt,
            seed=seed,
            callback_url=callback_url,
            execution=runtime.execution_options(),
        ),
    )


@audio_app.command("transcribe")
def audio_transcribe(
    ctx: typer.Context,
    audio: Annotated[str, typer.Option("--audio")],
    language: str = "auto",
    hints: str = "",
    denoise: bool = True,
    wait_option: WaitOption = None,
    download: DownloadOption = None,
    output_json: OutputJsonOption = False,
    jq_expr: JqOption = None,
) -> None:
    """Start speech transcription for an audio file."""

    runtime = apply_common_options(
        ctx,
        wait_option=wait_option,
        download=download,
        output_json=output_json,
        jq_expr=jq_expr,
    )
    run_action(
        ctx,
        lambda: runtime.services().execute_audio_transcribe(
            audio=audio,
            language=language,
            hints=hints,
            denoise=denoise,
            execution=runtime.execution_options(),
        ),
    )


@audio_app.command("tts-create")
def tts_create(
    ctx: typer.Context,
    text: Annotated[Optional[str], typer.Option("--text")] = None,
    text_file: Annotated[Optional[Path], typer.Option("--text-file")] = None,
    speaker: str = "Ryan",
    style: str = "Auto",
    language: str = "Auto",
    prompt: str = "",
    seed: int = -1,
    wait_option: WaitOption = None,
    download: DownloadOption = None,
    output_json: OutputJsonOption = False,
    jq_expr: JqOption = None,
) -> None:
    """Create speech from text using a selected speaker."""

    runtime = apply_common_options(
        ctx,
        wait_option=wait_option,
        download=download,
        output_json=output_json,
        jq_expr=jq_expr,
    )
    run_action(
        ctx,
        lambda: runtime.services().execute_tts_create(
            text=text,
            text_file=text_file,
            speaker=speaker,
            style=style,
            language=language,
            prompt=prompt,
            seed=seed,
            execution=runtime.execution_options(),
        ),
    )


@audio_app.command("tts-clone")
def tts_clone(
    ctx: typer.Context,
    audio: Annotated[str, typer.Option("--audio")],
    text: Annotated[Optional[str], typer.Option("--text")] = None,
    text_file: Annotated[Optional[Path], typer.Option("--text-file")] = None,
    language: str = "Auto",
    seed: int = -1,
    wait_option: WaitOption = None,
    download: DownloadOption = None,
    output_json: OutputJsonOption = False,
    jq_expr: JqOption = None,
) -> None:
    """Create speech from text using a cloned voice sample."""

    runtime = apply_common_options(
        ctx,
        wait_option=wait_option,
        download=download,
        output_json=output_json,
        jq_expr=jq_expr,
    )
    run_action(
        ctx,
        lambda: runtime.services().execute_tts_clone(
            audio=audio,
            text=text,
            text_file=text_file,
            language=language,
            seed=seed,
            execution=runtime.execution_options(),
        ),
    )


@audio_app.command("tts-voice")
def tts_voice(
    ctx: typer.Context,
    text: Annotated[Optional[str], typer.Option("--text")] = None,
    text_file: Annotated[Optional[Path], typer.Option("--text-file")] = None,
    character: str = "Female",
    style: str = "Auto",
    language: str = "Auto",
    prompt: str = "",
    seed: int = -1,
    wait_option: WaitOption = None,
    download: DownloadOption = None,
    output_json: OutputJsonOption = False,
    jq_expr: JqOption = None,
) -> None:
    """Create speech from text using a character voice."""

    runtime = apply_common_options(
        ctx,
        wait_option=wait_option,
        download=download,
        output_json=output_json,
        jq_expr=jq_expr,
    )
    run_action(
        ctx,
        lambda: runtime.services().execute_tts_voice(
            text=text,
            text_file=text_file,
            character=character,
            style=style,
            language=language,
            prompt=prompt,
            seed=seed,
            execution=runtime.execution_options(),
        ),
    )


@app.command("status")
def status(
    ctx: typer.Context,
    task_id: str,
    download: DownloadOption = None,
    output_json: OutputJsonOption = False,
    jq_expr: JqOption = None,
) -> None:
    """Show task status once; --download waits for completion first."""

    runtime = apply_common_options(
        ctx, download=download, output_json=output_json, jq_expr=jq_expr
    )

    def action() -> Any:
        return runtime.services().get_task_status(
            task_id=task_id,
            execution=runtime.execution_options(wait=False),
        )

    run_action(ctx, action)


@app.command("wait")
def wait_command(
    ctx: typer.Context,
    task_id: str,
    download: DownloadOption = None,
    output_json: OutputJsonOption = False,
    jq_expr: JqOption = None,
) -> None:
    """Wait for an existing async task to finish."""

    runtime = apply_common_options(
        ctx, download=download, output_json=output_json, jq_expr=jq_expr
    )
    run_action(
        ctx,
        lambda: runtime.services().wait_for_task(
            task_id=task_id,
            execution=runtime.execution_options(wait=True),
        ),
    )


@app.command("keywords")
def keywords(
    ctx: typer.Context,
    image: Annotated[str, typer.Option("--image")],
    lang: str = "en",
    num_keywords: Optional[int] = None,
    colors: bool = False,
    output_json: OutputJsonOption = False,
    jq_expr: JqOption = None,
) -> None:
    """Extract keywords from an image."""

    runtime = apply_common_options(ctx, output_json=output_json, jq_expr=jq_expr)
    run_action(
        ctx,
        lambda: runtime.services().execute_keywords(
            image=image,
            lang=lang,
            num_keywords=num_keywords,
            colors=colors,
        ),
    )


@app.command("quality")
def quality(
    ctx: typer.Context,
    image: Annotated[str, typer.Option("--image")],
    output_json: OutputJsonOption = False,
    jq_expr: JqOption = None,
) -> None:
    """Score technical image quality."""

    runtime = apply_common_options(ctx, output_json=output_json, jq_expr=jq_expr)
    run_action(
        ctx,
        lambda: runtime.services().execute_quality(image=image),
    )


@app.command("quality-ugc")
def quality_ugc(
    ctx: typer.Context,
    image: Annotated[str, typer.Option("--image")],
    output_json: OutputJsonOption = False,
    jq_expr: JqOption = None,
) -> None:
    """Score UGC image quality."""

    runtime = apply_common_options(ctx, output_json=output_json, jq_expr=jq_expr)
    run_action(
        ctx,
        lambda: runtime.services().execute_quality_ugc(image=image),
    )


@app.command("faces")
def faces(
    ctx: typer.Context,
    image: Annotated[str, typer.Option("--image")],
    output_json: OutputJsonOption = False,
    jq_expr: JqOption = None,
) -> None:
    """Detect faces in an image."""

    runtime = apply_common_options(ctx, output_json=output_json, jq_expr=jq_expr)
    run_action(ctx, lambda: runtime.services().execute_faces(image=image))


@app.command("captioning")
def captioning(
    ctx: typer.Context,
    image: Annotated[str, typer.Option("--image")],
    output_json: OutputJsonOption = False,
    jq_expr: JqOption = None,
) -> None:
    """Generate an image caption."""

    runtime = apply_common_options(ctx, output_json=output_json, jq_expr=jq_expr)
    run_action(
        ctx,
        lambda: runtime.services().execute_captioning(image=image),
    )


@app.command("video-keywords")
def video_keywords(
    ctx: typer.Context,
    video: Annotated[str, typer.Option("--video")],
    output_json: OutputJsonOption = False,
    jq_expr: JqOption = None,
) -> None:
    """Extract keywords from a video."""

    runtime = apply_common_options(ctx, output_json=output_json, jq_expr=jq_expr)
    run_action(
        ctx,
        lambda: runtime.services().execute_video_keywords(video=video),
    )


@app.command("run")
def generic_run(
    ctx: typer.Context,
    endpoint: str,
    prompt: Annotated[Optional[str], typer.Option("--prompt", "-p")] = None,
    item: Annotated[
        list[str],
        typer.Option(
            "--input",
            "-i",
            help="key=value input. Value is parsed as JSON when possible.",
        ),
    ] = [],
    input_file: Annotated[Optional[Path], typer.Option("--input-file")] = None,
    method: Annotated[Optional[str], typer.Option("--method")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    help_schema: Annotated[bool, typer.Option("--help-schema")] = False,
    wait_option: WaitOption = None,
    download: DownloadOption = None,
    output_json: OutputJsonOption = False,
    jq_expr: JqOption = None,
) -> None:
    """Call an endpoint by name/path with a payload from CLI options."""

    runtime = apply_common_options(
        ctx,
        wait_option=wait_option,
        download=download,
        output_json=output_json,
        jq_expr=jq_expr,
    )

    def action() -> Any:
        return runtime.services().execute_generic(
            endpoint=endpoint,
            payload=parse_generic_payload(
                prompt=prompt, items=item, input_file=input_file
            ),
            method=method,
            execution=runtime.execution_options(),
            dry_run=dry_run,
            help_schema=help_schema,
            client_id_present=runtime.client_id is not None,
            schema_loader=load_schema,
        )

    run_action(ctx, action)


def apply_local_output_options(
    ctx: typer.Context,
    output_json: bool,
    jq_expr: str | None,
) -> None:
    """Apply command-local output flags."""

    apply_common_options(ctx, output_json=output_json, jq_expr=jq_expr)


@mcp_app.command("serve")
def mcp_serve(ctx: typer.Context) -> None:
    """Start the Everypixel MCP server using the stdio transport."""

    runtime = get_runtime(ctx)
    from .mcp_server import create_mcp_server, run_mcp_server

    server = create_mcp_server(
        runtime.services,
        client_id_present=runtime.client_id is not None,
    )
    run_mcp_server(server)


@docs_app.command("openapi")
def docs_openapi(
    ctx: typer.Context,
    output_json: Annotated[bool, typer.Option("--output-json", "-j")] = False,
    jq_expr: Annotated[Optional[str], typer.Option("--jq")] = None,
) -> None:
    """Print the OpenAPI schema selected through live/cache/bundled fallback."""

    apply_local_output_options(ctx, output_json, jq_expr)
    runtime = get_runtime(ctx)
    run_action(ctx, lambda: runtime.services().openapi())


@docs_app.command("open")
def docs_open(ctx: typer.Context) -> None:
    """Open API Swagger UI in a browser."""

    runtime = get_runtime(ctx)

    def action() -> dict[str, Any]:
        url = f"{runtime.base_url.rstrip('/')}/v1/docs"
        webbrowser.open(url)
        return {"status": "ok", "url": url}

    run_action(ctx, action)


@docs_app.command("refresh")
def docs_refresh(ctx: typer.Context) -> None:
    """Force-refresh the local OpenAPI schema cache."""

    runtime = get_runtime(ctx)

    def action() -> dict[str, Any]:
        result, path = runtime.services().refresh_openapi()
        schema = serialize_operation_result(result)
        return {
            "status": "ok",
            "source": "live",
            "cache_path": str(path),
            "paths": len(schema.get("paths", {})),
        }

    run_action(ctx, action)


@schema_app.command("refresh")
def schema_refresh(ctx: typer.Context) -> None:
    """Alias for `docs refresh`."""

    docs_refresh(ctx)


@schema_app.command("openapi")
def schema_openapi(
    ctx: typer.Context,
    output_json: Annotated[bool, typer.Option("--output-json", "-j")] = False,
    jq_expr: Annotated[Optional[str], typer.Option("--jq")] = None,
) -> None:
    """Alias for `docs openapi`."""

    docs_openapi(ctx, output_json=output_json, jq_expr=jq_expr)


if __name__ == "__main__":
    app()
