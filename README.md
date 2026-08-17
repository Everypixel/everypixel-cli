# Everypixel CLI

Official command-line interface for the [Everypixel API](https://labs.everypixel.com/).
Generate and edit images and videos, work with audio, run visual analysis, and
automate Everypixel workflows from a terminal or an MCP client.

The CLI is designed for both people and agents: commands use readable terminal
output by default and can return stable JSON for scripts and tool integrations.

- [Everypixel Labs](https://labs.everypixel.com/)
- [API documentation](https://labs.everypixel.com/docs)
- [GitHub repository](https://github.com/everypixel/everypixel-cli)
- [Issue tracker](https://github.com/everypixel/everypixel-cli/issues)

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

Create API credentials in your Everypixel Labs account, then save them in a
local CLI profile:

```bash
everypixel auth configure \
  --client-id "$EVERYPIXEL_CLIENT_ID" \
  --client-secret "$EVERYPIXEL_CLIENT_SECRET"

everypixel auth check
```

Generate an image and download the completed result:

```bash
everypixel image generate \
  "editorial product photo on a white background" \
  --model flux2 \
  --size square \
  --download ./outputs
```

The same command can return JSON for automation:

```bash
everypixel image generate \
  "editorial product photo on a white background" \
  --model flux2 \
  --no-wait \
  --output-json
```

Use `--help` on any command to see all supported models and options:

```bash
everypixel image generate --help
everypixel video edit --help
```

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

everypixel image angles \
  --image ./input.png \
  --azimuth right \
  --elevation eye_level \
  --distance medium \
  --download ./outputs
```

### Video

| Command | Description |
| --- | --- |
| `everypixel video generate PROMPT` | Generate video from text |
| `everypixel video from-image PROMPT --image FILE` | Animate an image |
| `everypixel video first-last PROMPT` | Generate video between first and last frames |
| `everypixel video edit` | Edit video with text and media references |
| `everypixel video upscale` | Upscale a video, URL, or existing task result |

Examples:

```bash
everypixel video generate \
  "a slow aerial shot over mountain lakes" \
  --model ltx23 \
  --duration 10 \
  --resolution 720p \
  --download ./outputs

everypixel video from-image \
  "subtle camera movement and natural wind" \
  --image ./first-frame.png \
  --model ltx23 \
  --download ./outputs

everypixel video edit \
  "make the scene cinematic" \
  --video ./input.mp4 \
  --download ./outputs
```

### Lipsync and Audio

| Command | Description |
| --- | --- |
| `everypixel lipsync video` | Synchronize a video with an audio track |
| `everypixel lipsync image` | Create a lipsync video from a still image and audio |
| `everypixel audio transcribe` | Transcribe speech from an audio file |
| `everypixel audio tts-create` | Create speech with a built-in speaker |
| `everypixel audio tts-clone` | Create speech from a cloned voice sample |
| `everypixel audio tts-voice` | Create speech with a character voice |

Examples:

```bash
everypixel lipsync video \
  --video ./input.mp4 \
  --audio ./voice.mp3 \
  --download ./outputs

everypixel audio transcribe \
  --audio ./speech.mp3 \
  --language English \
  --download ./outputs

everypixel audio tts-create \
  --text "Hello world" \
  --speaker Ryan \
  --style Warm \
  --language English \
  --download ./outputs
```

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

A Bash automation example:

```bash
task_id="$(
  everypixel image generate \
    "automation test image" \
    --model flux2 \
    --no-wait \
    --output-json \
    --jq .task_id | tr -d '"'
)"

everypixel --timeout 600 wait "$task_id" --download ./outputs
```

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
| `--timeout SECONDS` | Set the polling timeout |
| `--poll-interval SECONDS` | Set the delay between status requests |
| `--download DIRECTORY`, `-d DIRECTORY` | Download completed result files |
| `--no-color` | Disable colored terminal output |
| `--debug` | Include sanitized diagnostic locations for internal errors |

Many operation-specific options can also appear after their command. Run
`everypixel COMMAND --help` for the exact syntax.

## Configuration Profiles

Profiles keep credentials and settings separate for different accounts or
projects. Most users can stay on the `default` profile.

```bash
everypixel config profile create project-a
everypixel config profile use project-a
everypixel auth configure \
  --client-id "$PROJECT_A_CLIENT_ID" \
  --client-secret "$PROJECT_A_CLIENT_SECRET"

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
  `everypixel --timeout 600 wait TASK_ID`.
- Invalid media: pass an existing local file, an HTTP(S) URL, or a supported
  data URI.
- File not found: use an absolute path or run the command from the directory
  containing the file.
- Download failure: verify that the output directory is writable and retry with
  `everypixel wait TASK_ID --download ./outputs`.
- No color: pass `--no-color` or set `NO_COLOR=1`.

## License

[MIT](LICENSE)
