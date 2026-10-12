# Everypixel CLI

Official command-line interface for the [Everypixel API](https://labs.everypixel.com/).
Generate and edit images and videos, chat with GLM-5.3, work with audio, run
visual analysis, and automate Everypixel workflows from a terminal or an MCP client.

Commands return readable terminal output by default and JSON for scripts.
The package includes a stdio MCP server for agent integrations.

[API documentation](https://labs.everypixel.com/docs) ·
[GitHub](https://github.com/everypixel/everypixel-cli) ·
[Issues](https://github.com/everypixel/everypixel-cli/issues) ·
[Changelog](CHANGELOG.md)

- [Quick start](#quick-start)
- [Images](#images), [video](#video), [audio](#lipsync-and-audio), [chat](#chat), [analysis](#visual-analysis)
- [Tasks and downloads](#async-tasks-and-downloads)
- [JSON output](#json-output) and [generic API calls](#generic-openapi-runner)
- [Authentication](#authentication) and [profiles](#configuration-profiles)
- [MCP server](#mcp-server)
- [Global options](#global-options), [exit codes](#exit-codes), [troubleshooting](#troubleshooting)

## Installation

Everypixel CLI requires Python 3.10 or newer.

```bash
pip install everypixel-cli
```

The package installs two equivalent commands:

```bash
everypixel --help
epx --help
```

## Quick Start

Create API credentials in your [Everypixel Labs account](https://labs.everypixel.com/).
Run the interactive setup and enter the client ID and secret when prompted:

```bash
everypixel auth configure
everypixel auth check
```

Generate an image and save it locally:

```bash
everypixel image generate "editorial product photo on a white background" \
  --model flux3 --download ./outputs
```

Media commands wait for completion by default. See [Tasks and downloads](#async-tasks-and-downloads)
for background jobs and [JSON output](#json-output) for scripts.

Use `--help` for command syntax. To list image generation models, select their
enum from OpenAPI:

```bash
everypixel image generate --help
everypixel docs openapi --jq .components.schemas.ImageGenerateModelEnum.enum
```

Examples use Bash syntax. Examples that extract IDs with `jq` require the
separate `jq` command.

## Authentication

Everypixel uses HTTP Basic Auth with the client ID as the username and the
client secret as the password. You can save credentials with `auth configure`
or provide them through environment variables:

```bash
export EVERYPIXEL_CLIENT_ID="your-client-id"
export EVERYPIXEL_CLIENT_SECRET="your-client-secret"
```

| Variable | Purpose |
| --- | --- |
| `EVERYPIXEL_CLIENT_ID` | API client ID |
| `EVERYPIXEL_CLIENT_SECRET` | API client secret |
| `EVERYPIXEL_BASE_URL` | Override the API base URL |
| `EVERYPIXEL_PROFILE` | Select a saved configuration profile |
| `NO_COLOR` | Disable colored terminal output when set |

Explicit command-line options take priority over environment variables, and
environment variables take priority over saved configuration.

Useful authentication commands:

| Command | Description |
| --- | --- |
| `everypixel auth configure` | Save credentials for the current profile |
| `everypixel auth check` | Verify the configured credentials |
| `everypixel auth whoami` | Show the active profile and credential source |
| `everypixel auth logout` | Remove credentials from the current profile |

Secrets are stored in the operating system keyring when one is available. The
CLI falls back to its user configuration file when a keyring is unavailable;
`everypixel config list` masks saved secrets in its output.

## Commands

### Images

| Command | Description |
| --- | --- |
| `everypixel image generate PROMPT` | Generate an image from a text prompt |
| `everypixel image edit PROMPT --image FILE` | Edit one or more images |
| `everypixel image vectorize FILE` | Convert an image to SVG |
| `everypixel image upscale` | Upscale an image, URL, or existing task result |
| `everypixel image angles --image FILE` | Change the camera angle of an image |
| `everypixel image colors --image FILE --reference FILE` | Transfer colors from a reference image |

Examples:

```bash
everypixel image edit \
  "replace the background with a clean studio wall" \
  --image ./input.png \
  --model qwen \
  --download ./outputs

everypixel image upscale --image ./input.png --download ./outputs

everypixel image generate "a mountain icon" \
  --model recraftv4_1_vector \
  --controls '{"colors":[{"rgb":[20,100,180],"weight":1}]}' \
  --download ./outputs

everypixel image vectorize ./input.png --download ./outputs

everypixel image angles \
  --image ./input.png \
  --azimuth right \
  --elevation eye_level \
  --distance medium \
  --download ./outputs
```

Recraft V4.1 vector models (`recraftv4_1_vector`, `recraftv4_1_pro_vector`)
produce SVG files and accept optional JSON palette `--controls`. Use `--size`
to select their aspect ratio. Raster models also accept `--resolution`.

`--lora-url` is supported only by `zimage`.

GPT Image models are `gpt-image-2` and `gpt-image-2.5-sunburst`. Both generation
and editing accept `--quality low|medium|high` (default: `medium`); Sunburst also
supports `xhigh` and `max`. Use `--size square` with `--resolution 3k`.

```bash
everypixel image generate "a detailed botanical illustration" \
  --model gpt-image-2.5-sunburst --quality high --download ./outputs
```

Other image models include `flux3`, `gemini-nano-banana-2.1` (Nano Banana 2.1),
`gemini-3.1-flash-lite` (Nano Banana 2 Lite), and `seedream-5-flash`.
Nano Banana 2 Lite uses `--resolution 1k`. Repeat `--image` to provide multiple
inputs to an editing model.

### Video

| Command | Description |
| --- | --- |
| `everypixel video generate PROMPT` | Generate video from text |
| `everypixel video from-image PROMPT --image FILE` | Animate an image |
| `everypixel video first-last PROMPT` | Generate video between first and last frames |
| `everypixel video edit` | Edit video with text and media references |
| `everypixel video upscale` | Upscale a video, URL, or existing task result |

Animate a local image or continue a video:

```bash
everypixel video from-image "subtle camera movement and natural wind" \
  --image ./first-frame.png --model ltx23 --download ./outputs

everypixel video edit "continue the scene with a slow camera pan" \
  --model flux3 --video ./input.mp4 --download ./outputs
```

Generate a video with Gemini Omni 1.1 Flash, then edit the completed task:

```bash
task_id="$(
  everypixel video generate "a slow aerial shot over mountain lakes" \
    --model gemini-omni-1.1-flash --output-json | jq -er '.task_id'
)"

everypixel video edit "change the scene to sunset" \
  --model gemini-omni-1.1-flash --task-id "$task_id" --download ./outputs
```

Gemini Omni editing requires an owned Gemini Omni source task. Other editing
models accept `--task-id` as an alternative to `--video`.

Input options depend on the model:

- `first-last` accepts `--image` and `--last-image` for models that support frame pairs.
- MiniMax and WAN 3.0 accept repeatable reference options, separately from frame inputs.
  Use `--reference-image`, `--reference-video`, and, for MiniMax, `--reference-audio`.
- Gemini Omni generation accepts frame images or `--reference-image`, separately.
- Grok Imagine 1.5 (`grok-imagine-1.5`) requires `--image` and supports 1080p.
- Aleph 2 accepts repeatable `--keyframe` JSON; inspect `video edit --help` for its fields.

See the [API documentation](https://labs.everypixel.com/docs) for each model's
accepted inputs, resolutions, and durations.

For upscaling, use `seedvr2` (the default) or a Topaz model such as `topaz-prob-4`:

```bash
everypixel video upscale --video ./input.mp4 \
  --model topaz-prob-4 --resolution 4k --download ./outputs
```

### Lipsync and Audio

| Command | Description |
| --- | --- |
| `everypixel lipsync video` | Synchronize a video with an audio track |
| `everypixel lipsync image` | Create a lipsync video from a still image and audio |
| `everypixel audio transcribe` | Transcribe speech from an audio file |
| `everypixel audio tts-create` | Create speech with a Qwen preset or saved ElevenLabs voice |
| `everypixel audio tts-clone` | Create Qwen speech from an audio sample or saved voice |
| `everypixel audio tts-design` | Design voice previews from a description |
| `everypixel audio tts-voice` | Save audio samples or a design preview as a voice |
| `everypixel audio tts-voices` | List presets and saved voices |
| `everypixel audio tts-delete` | Delete an owned saved voice |

Synchronize video, transcribe audio, or synthesize speech with a Qwen preset:

```bash
everypixel lipsync video --video ./input.mp4 --audio ./voice.mp3 \
  --download ./outputs

everypixel audio transcribe --audio ./speech.mp3 --language English \
  --download ./outputs

everypixel audio tts-create --model qwen3 --text "Hello world" \
  --speaker Ryan --style Warm --language English --download ./outputs
```

`--speaker`, `--style`, `--language`, and `--seed` are Qwen controls;
`--character` applies only to Qwen voice design. ElevenLabs synthesis uses
`--voice-id` instead of preset controls. `tts-clone` and `tts-design` default
to Qwen; `tts-create` lets the API choose when `--model` is omitted.

Design an ElevenLabs preview, save the first preview as a voice, and use it:

```bash
preview_id="$(
  everypixel audio tts-design --model eleven_ttv_v3 \
    --prompt "A warm, calm narrator with clear diction and a relaxed pace." \
    --text "Welcome to our guided tour. Today we will explore the stories behind these photographs and the people who made them." \
    --output-json | jq -er '.result.previews[0].preview_id'
)"

voice_id="$(
  everypixel audio tts-voice --name "Narrator" --preview-id "$preview_id" \
    --output-json | jq -er '.voice.id'
)"

everypixel audio tts-create --model eleven_v3 --voice-id "$voice_id" \
  --text "Hello world" --download ./outputs
```

To save your own samples, use `tts-voice --name "My voice" --provider elevenlabs
--audio ./sample.wav`; repeat `--audio` for more samples. Qwen uses
`--provider labs` and one sample. Saved Qwen voices work with `tts-clone --voice-id`.

List saved ElevenLabs voices with `everypixel audio tts-voices --provider elevenlabs`.
See the [API documentation](https://labs.everypixel.com/docs) for model choices
and text constraints.

### Chat

```bash
everypixel chat "Explain how a rainbow forms."
everypixel chat "Review this idea." --system "Be concise." --stream
everypixel chat "Solve this problem." --reasoning-effort high --show-reasoning
everypixel chat --input-file ./conversation.json --output-json
```

Chat uses the same credentials as other commands and requires the client's
`chat` scope. The default model is `glm-5.3`. Each invocation sends one request;
use `--input-file` for conversation history, function tools, or other API options:

```json
{
  "messages": [
    {"role": "user", "content": "What causes rainbows?"},
    {"role": "assistant", "content": "Sunlight refracts and reflects in water droplets."},
    {"role": "user", "content": "Why do the colors separate?"}
  ],
  "max_completion_tokens": 4096
}
```

A positional prompt appends a user message to the supplied history. CLI options
override matching fields in the file. `--stream` displays text as it arrives;
JSON output returns one assembled response, including reasoning, tool calls,
and usage. Function calls are returned for the caller to handle. The MCP `chat`
tool accepts the same structured request and returns a complete response.
Use `--request-timeout` to change the chat HTTP timeout (default: 240 seconds).

### Visual Analysis

Analysis commands accept HTTP(S) URLs and local files. Local files are uploaded
automatically when the endpoint supports them.

| Command | Description |
| --- | --- |
| `everypixel keywords --image FILE` | Extract image keywords |
| `everypixel quality --image FILE` | Score technical image quality |
| `everypixel quality-ugc --image FILE` | Score user-generated image quality |
| `everypixel faces --image FILE` | Detect faces in an image |
| `everypixel captioning --image FILE` | Generate an image caption |
| `everypixel video-keywords --video FILE` | Extract video keywords |

```bash
everypixel keywords \
  --image ./image.jpg \
  --lang en \
  --num-keywords 10 \
  --colors \
  --output-json

everypixel captioning --image ./image.jpg --output-json
```

## Async Tasks and Downloads

Generation and media-processing commands wait for completion by default. Add
`--no-wait` to return the newly created task immediately.

```bash
everypixel image generate "minimalist poster" --no-wait --output-json
```

Use `status` for one status request and `wait` to poll until the task reaches a
terminal state:

```bash
everypixel status TASK_ID
everypixel wait TASK_ID
```

Passing `--download` always waits for a successful result before saving files:

```bash
everypixel wait TASK_ID --download ./outputs
```

`elapsed_sec` is total local waiting time; `queue_sec` estimates time until the
first observed `STARTED` response.

API costs are USD strings: `estimated_cost` for pending tasks and `billed_cost`
for successful results and chat. Missing costs are omitted; `"0"` means no charge.

## Generic OpenAPI Runner

`everypixel run` calls an API operation by its OpenAPI name or `/v1/...` path.
Values passed with `--input` are parsed as JSON when possible. The CLI validates
named operations against the selected OpenAPI schema before sending a request.

```bash
everypixel run image_generate \
  --prompt "product photo" \
  --input model=flux2 \
  --input image_size=square \
  --dry-run

everypixel run image_generate --help-schema --output-json

everypixel run /v1/image_generate \
  --method POST \
  --input-file ./payload.json
```

`--help-schema` shows the operation definition. Referenced component schemas
are included in the full document returned by `docs openapi`.

The schema loader tries the live API first, then the local cache, and finally
the schema bundled with the package.

```bash
everypixel docs open
everypixel docs openapi --output-json
everypixel docs refresh
```

## JSON Output

Use `--output-json` (or `-j`) for scripts and `--jq` to select part of the
result:

```bash
everypixel keywords --image ./image.jpg --output-json
everypixel keywords --image ./image.jpg --output-json --jq .keywords
```

Handled successes and errors are written as one machine-readable JSON stream to
stdout. Human-readable errors are written to stderr. Add `--debug` when an
unexpected internal error needs sanitized stack-frame locations.

## Global Options

Global options must appear before the command they apply to.

| Option | Description |
| --- | --- |
| `--base-url URL` | Override the API base URL for one invocation |
| `--profile NAME` | Use a saved configuration profile |
| `--output human\|json` | Select the output format |
| `--output-json`, `-j` | Shortcut for JSON output |
| `--jq EXPRESSION` | Filter JSON output |
| `--wait`, `--no-wait` | Control async task polling |
| `--timeout SECONDS` | Set the polling timeout (default: 30 minutes) |
| `--poll-interval SECONDS` | Set the delay between status requests |
| `--download DIRECTORY`, `-d DIRECTORY` | Download completed result files |
| `--no-color` | Disable colored terminal output |
| `--debug` | Include sanitized diagnostic locations for internal errors |

Operation commands also accept common options such as `--download`, `--no-wait`,
and `--output-json` after the command. Run `everypixel COMMAND --help` for the
exact syntax.

## Configuration Profiles

Profiles keep credentials and settings separate for different accounts or
projects. Most users can stay on the `default` profile.

```bash
everypixel config profile create project-a
everypixel config profile use project-a
everypixel auth configure
everypixel config list
```

Select a profile for one invocation without changing the active profile:

```bash
everypixel --profile project-a auth check
```

## MCP Server

Everypixel CLI includes an MCP server for Claude Desktop, Claude Code, Cursor,
VS Code, and other clients that support MCP over stdio:

```bash
everypixel mcp serve
```

Example client configuration:

```json
{
  "mcpServers": {
    "everypixel": {
      "command": "everypixel",
      "args": ["mcp", "serve"],
      "env": {
        "EVERYPIXEL_CLIENT_ID": "your-client-id",
        "EVERYPIXEL_CLIENT_SECRET": "your-client-secret"
      }
    }
  }
}
```

The MCP client launches the command as a subprocess; no HTTP port is opened.
Local media and download paths use the subprocess working directory and
filesystem permissions.

## Shell Completion

Install completion for the current shell:

```bash
everypixel --install-completion
```

Or print the completion script for manual setup:

```bash
everypixel --show-completion
```

## Exit Codes

| Code | Meaning |
| --- | --- |
| `0` | Success |
| `1` | General error |
| `2` | Local argument or validation error |
| `3` | Configuration or authentication error |
| `4` | API, remote service, or task failure |
| `5` | File read/write or result download error |
| `6` | Result processing, jq, or serialization error |

## Troubleshooting

- Authentication failure: run `everypixel auth check`, then configure the
  credentials again if needed.
- Wait timeout: increase the global timeout, for example
  `everypixel --timeout 7200 wait TASK_ID`.
- Invalid media: pass an existing local file, an HTTP(S) URL, or a supported
  data URI.
- File not found: use an absolute path or run the command from the directory
  containing the file.
- Download failure: verify that the output directory is writable and retry with
  `everypixel wait TASK_ID --download ./outputs`.
- No color: pass `--no-color` or set `NO_COLOR=1`.

## License

[MIT](LICENSE)
