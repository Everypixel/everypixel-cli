# Changelog

## [0.2.2] - 2026-09-11

- Increase the default task wait timeout to 30 minutes for long video generation.

## [0.2.1] - 2026-09-11

- Support up to 9 image, 3 video, and 3 audio references for MiniMax H3/Turbo.

## [0.2.0] - 2026-09-09

- Add MiniMax H3/Turbo, WAN 3.0, Flux 3, and Seedance 2.5 video models.
- Add Recraft vector generation, palette controls, and image-to-SVG conversion.
- Update Grok Imagine models, image-edit input limits, and megapixel controls.
- Add Topaz 4K video upscaling and 720p lipsync.

## [0.1.0] - 2026-08-17

Initial public release.

### Added

- Image generation with `zimage`, `wan22`, `wan2.7`, `wan2.7-pro`, `flux2`,
  `grok`, `grok_quality`, `gemini-3.1-flash`, `gemini-3-pro`,
  `seedream-5-pro`, `seedream-5`, and GPT Image 2 low, medium, and high models.
- Image editing with `flux2`, `qwen`, `wan2.7`, `wan2.7-pro`, `grok`,
  `grok_quality`, Gemini 3, Seedream 5, and GPT Image 2 model families.
- Image upscaling with `seedvr2`, camera-angle generation, and color transfer.
- Video generation with `wan22`, `ltx23`, `grok`, `grok15`, `veo-3.1`,
  `veo-3.1-fast`, `seedance2`, `seedance2-mini`, `kling-2.6`, `kling-3`,
  `kling-3-turbo`, `kling-3-omni`, and `wan2.7`.
- Text-to-video, image-to-video, first/last-frame generation, video editing,
  and video upscaling to 720p, 1080p, or 1440p.
- Video editing with `seedance2`, `seedance2-mini`, `kling-3-omni`, `wan2.7`,
  and `aleph2`, including provider-specific media and keyframe inputs.
- Video and image lipsync, with `inftalk` and `ltx23` available for image
  animation.
- Audio transcription, text-to-speech with built-in speakers, voice cloning,
  and character voice generation.
- Image keywords and colors, image quality and UGC quality scores, face
  analysis, image captioning, and video keywords.
- Shared asynchronous task status, polling, timeout, cancellation, and streamed
  result downloads with safe atomic filenames.
- Local file, HTTP(S) URL, and data URI media inputs with payload validation
  before network requests.
- Human-readable and JSON output, `--jq` filtering, stable exit codes, and
  sanitized debug diagnostics.
- Named profiles, development-environment selection, environment overrides,
  Basic Auth checks, and keyring-backed credential storage.
- Generic `run` support for OpenAPI operations with live, cached, and bundled
  schemas plus local `$ref`, composed-schema, and constraint validation.
- A stdio MCP server exposing the same image, video, audio, task, analysis,
  authentication, and OpenAPI workflows as 26 structured tools.
- `everypixel` and `epx` command entry points for Python 3.10 and newer.
