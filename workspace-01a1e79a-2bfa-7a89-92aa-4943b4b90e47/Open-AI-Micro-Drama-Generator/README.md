# MicroDrama AI — saved projects and flexible media

A local Next.js + FastAPI app that turns an idea or scene script into a short video. For each job, choose **simple, Google Gemini, OpenAI GPT, or an OpenAI-compatible LLM for story and shot planning**; choose **Gemini API or A2E for images**; and choose an **A2E image-to-video model**. MoviePy combines the resulting clips. The original MuAPI media integration has been replaced with the supplied A2E API contract.

> **Generation can cost money.** A job normally creates multiple text-planning requests (if enabled), images and video clips. Google bills Gemini text and/or images separately from A2E; OpenAI GPT and a configured OpenAI-compatible LLM may also charge separately. A2E quotes are estimates, not a price guarantee or reservation, and the A2E limit does not cover text or Gemini costs. OpenAI and Google may bill a request even if a later step fails. For Gemini frames, A2E also charges for issuing each upload URL—even if the subsequent upload fails. Review your providers' current prices and account limits before running a job.

**Testing status:** The integration, payloads, model routing, upload flow and UI were tested with mocks and an isolated browser/backend. No live, paid OpenAI, Gemini or A2E generation has been performed. Model access, API contracts, pricing and availability can change; a listed model is **not** a guarantee that your account can use it.

## How it works

1. Pick **Idea to Video** or **Script to Video** and a planning provider. **Simple** uses your supplied text directly and plans simple shots—it does not invent an expanded story or extract characters. **Gemini** uses your saved Gemini key and the selected text model to develop the story, write scenes, extract characters and plan shots. In script mode it extracts characters and plans shots from your supplied script, without rewriting it. **OpenAI GPT** does the same planning through the OpenAI Responses API with its own saved API key and selected GPT text model. **OpenAI-compatible** uses the separate chat-completions LLM configured in `server/.env`. All scenes and storyboards are planned and saved before any paid image or video requests, so a planning failure cannot waste credits on portraits. AI planning may itself be billed and can produce extra scenes or character portraits. Planning and image providers are independent per-job choices: GPT planning works with A2E or Gemini images. Saving any key does not select a planner automatically.
2. Choose an image provider and model for the job. A2E images use the selected A2E image endpoint. Gemini images use the same saved Google key through the Gemini Interactions API directly. Previous frames or character portraits can be reused as image references for continuity where supported; consistency is not guaranteed.
3. When using Gemini images, each shot frame is uploaded to A2E staging storage using a paid presigned URL; A2E receives its public CDN URL. Character-only Gemini portraits are saved locally as references and do not incur an A2E upload fee. Gemini-generated images include Google's SynthID watermark. Google is called with `store=false` for stateless interactions, subject to its applicable terms and data practices.
4. Choose an A2E video model and the duration **per clip** where supported. The chosen model uses the frame as either a first frame or a reference, as indicated in the UI. The app quotes each A2E generation automatically and submits it if the estimate fits the per-job guard. Completed clips are downloaded promptly and combined into an MP4.

### Models exposed in the website

The catalog reflects the supplied specification and provider model documentation, not live provider entitlements. It covers **6 GPT text choices**, **6 Gemini text choices**, **30 A2E image choices**, **5 direct Gemini image choices**, and **41 A2E video choices** including quality/version variants. The `GET /api/models` endpoint is the definitive list of the IDs this version accepts.

- **OpenAI GPT text planning:** GPT-6 Luna (default), GPT-6.1 Terra, GPT-6.1 Sol, GPT-6 Astra, GPT-5.4 mini and GPT-5.5. These use the OpenAI API directly for text only, not GPT Image through A2E. Account/model access, pricing and availability may change; a listed model is not an entitlement. The model is selected per job and is independent of image choice.
- **Gemini text planning:** Gemini 3.5 Flash-Lite, 3.6 Flash, 3.8 Flash (default), 4 Hydrogen, 4 Argon and 4 Carbon. Availability and billing depend on the Google project; listed models may not be available on every key. Text and image models are selected separately, and a job can use Gemini text with A2E images.
- **Gemini images directly:** Nano Banana 2.1 (`gemini-nano-banana-2.1`), Nano Banana 2 Lite (`gemini-3.1-flash-lite-image`), Nano Banana 2 (`gemini-3.1-flash-image`), Nano Banana Pro (`gemini-3-pro-image`), and legacy Nano Banana (`gemini-2.5-flash-image`). Google has deprecated Nano Banana 2 and recommends 2.1; it is offered for existing workflows while available. They require a Gemini key and Google model access. Lite is not optimized for multiple references or sequential editing. Requests normally use default ~1K image output; Google availability and prices may change.
- **Images through A2E:** A2E and A2E V2, Z-Image Turbo, Seedream, Nano Banana, GPT Image, Wan Image, Qwen Image, Flux, Grok Image and Kling Image variants supported by the included catalog. These use the **A2E key and A2E billing**, including when the model is called Nano Banana. Z-Image is text-only and cannot use reference images; the app skips portraits for that choice. GPT Image 1.5 uses a 3:2 landscape frame because it does not support 16:9.
- **Video through A2E:** native A2E, Wan (including Spicy, 3.0 and reference-to-video), HappyHorse, Veo, Kling (including Omni), Grok, Hailuo, MiniMax H3 and Seedance variants. Kling Omni and Wan 2.6 R2V use the generated image as a **reference**, not a guaranteed first frame. Modes requiring additional input video, an end frame, audio files or other assets are not exposed by this one-image workflow.

Durations depend on the selected model and are validated before the job starts. Veo's A2E endpoint exposes no duration control, so its clip length is provider-determined. For other models, the selected duration is for **each** shot; total runtime is roughly the sum of the clips. Native A2E original rounds requests over 5 seconds up to 5-second segments (for example, a 6-second request can produce about 10 seconds); V2 variants can run up to about 0.7 seconds longer due to frame rounding. The app presets resolution and other provider-specific settings; where implemented, quality/version variants appear as separate choices rather than a general resolution control. Wan 3.0 standard can be much slower than Prime, so longer jobs may need a larger `A2E_POLL_TIMEOUT_SECONDS`; timing out does not cancel the remote task. The integration requests audio off where the selected endpoint supports it, except that native A2E models follow `A2E_AUTO_ADD_AUDIO`. Some models include audio or do not expose an audio-off switch. Sound, dialogue and lip sync are not guaranteed.

## Quick start

### Upgrading an existing installation

Stop both servers and back up the existing private `server/.env`, `server/projects/` (if present) and `server/outputs/` (if present). Extract this version to a new folder, then copy those files/folders locally into its `server/` directory; do not share the `.env` or saved projects. The ZIP deliberately does not contain your keys, saved projects or generated outputs. Install/update backend requirements, run `npm ci` and `npm run build`, then start both servers as described below. Old jobs created before checkpoints were added cannot be resumed automatically, though an old MP4 may still be accessible if its output file exists.

### Prerequisites

- Python 3.10+ and Node.js 20.9+ / npm.
- An [A2E API token](https://video.a2e.ai/account/token), generally beginning `sk_`, plus credits and access to the selected A2E models.
- If using Gemini for text planning, images, or both, a [Gemini API key](https://aistudio.google.com/api-keys) with access to the selected text/image models and any required Google billing enabled. The same key serves both uses.
- If using GPT for planning, an [OpenAI Platform API key](https://platform.openai.com/api-keys) with access to the selected GPT model and any required API billing. OpenAI API billing is separate from a ChatGPT subscription.
- FFmpeg available to MoviePy. `imageio-ffmpeg`, installed with MoviePy, normally supplies a binary; some platforms require a separate FFmpeg installation.
- Optional: a separate OpenAI-compatible chat-completions key for the writer, character extractor and storyboard artist.

### Backend

From the extracted project directory:

```bash
cd server
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
[ -f .env ] || cp .env.example .env  # Windows cmd: if not exist .env copy .env.example .env
uvicorn api:app --host 127.0.0.1 --port 8000
```

The command above leaves an existing `.env` untouched. Keys can be saved from the website after startup, or entered in the private `server/.env` file. Restart the backend after manual `.env` edits; keys saved through the website take effect immediately for new jobs. `http://127.0.0.1:8000/api/health` reports configuration flags without returning key values.

### Frontend

In another terminal:

```bash
cd client
npm ci
npm run build
npm run start -- --hostname 127.0.0.1 --port 3000
```

Open <http://127.0.0.1:3000>. Save your A2E key and, if desired, your Gemini and/or OpenAI API keys in their respective fields. Saving does **not** verify a key with its provider; each field clears after saving. For GPT story writing, choose **OpenAI GPT API** under **Story & shot planning**, then choose a GPT text model. For Gemini, choose **Google Gemini API** and a Gemini text model. Independently select the image provider/model, A2E video model and supported clip duration. Review the charge notice and start a job. The progress log includes the selected planner, A2E estimates and task IDs.

The production build uses versioned JavaScript assets to avoid stale browser bundles. For local development, `npm run dev -- --hostname 127.0.0.1 --port 3000` is also available. Next.js proxies browser `/api/*` and `/outputs/*` requests to the backend. Set `BACKEND_URL` for the Next.js process **before building** if the backend is not at `http://127.0.0.1:8000`.

### Resume a failed or interrupted project

1. On the home page, open **Saved projects**, or revisit the original project URL. Completed planning steps, saved images, clips and any final video are available on that project's page. Use **View saved A2E task IDs** to check known tasks, including earlier attempts. A server restart marks an unfinished job **Interrupted**; it never automatically resumes or submits another paid request.
2. Check your provider task and billing history, especially if a submission response was lost or a task failed. A2E tasks are named with the first eight characters of the project ID, plus scene/shot details, where the endpoint supports names. The application cannot refund credits or guarantee that a provider can recover a lost response.
3. If needed, save a working key in Settings, then reopen the project. Its original planning/image/video selections are retained. A resume uses the currently configured keys, not the keys held by the previous process; keep the same A2E account for polling its existing tasks and use an OpenAI/Gemini key with access to the original planning model. Restore the original planning limits, A2E API origin, configured LLM model/origin and native-audio setting if they were changed. The A2E estimate limit may be increased separately if appropriate.
4. Confirm that remaining work can incur charges and select **Resume from saved work**. If any paid request has an unknown outcome, a terminal failure, or a provider-confirmed expired output, an additional explicit duplicate-charge confirmation is required. Review the listed steps and task IDs before agreeing. Previously completed steps and intact local files are reused; known unfinished A2E task IDs are polled, not submitted again.

Image URLs can expire or become inaccessible (HTTP 403, 404 or 410). If a saved image is needed for **new** work, the app can re-upload its verified local copy to A2E; the signing request may cost credits. A completed clip or existing video task does not require an input-image re-upload. An inaccessible local copy and an expired remote result may be unrecoverable; a replacement, when offered, requires explicit duplicate-charge consent. If a saved Gemini image is missing, restore its original file from a backup to reuse it without another Google image request, or explicitly consent to a replacement. A transient image-link check error does not trigger a paid re-upload. Cumulative A2E estimates, including prior attempts and upload-signing estimates, remain in the job's budget guard; they are not actual charges and do not cover Google, OpenAI or another text provider.

Projects created before this recovery feature have no checkpoints to migrate automatically. If their old files or task IDs are gone, check the provider's account history; this app cannot recover or refund them. Recovery also cannot guarantee results lost before a provider returned a task ID, lost Google responses, full-disk/corrupt files, or output already removed by a provider. Keep the `server/projects` folder on persistent storage (for containers, mount a persistent volume directly at that path; symlinked project directories are refused), back it up when the server is stopped, and maintain enough free space. In temporary/hosted environments, downloads, restarts or workspace snapshot limits may remove files; verify that storage is truly persistent.

### Security and privacy

This is a development app, **not an authenticated multi-user service**. Anyone who can access the UI or backend can replace keys, start chargeable jobs and view outputs. The A2E, Gemini and OpenAI keys are stored **unencrypted** in `server/.env`; permissions are restricted to the owner on POSIX systems. Key values are not returned to the browser, placed in browser storage or included in project downloads. Enter real keys only on a private, trusted instance; bind both servers to loopback as above. Do not expose the app publicly without authentication, authorization, CSRF protection, rate limits, private output storage and per-user spending controls. A custom request header is not authentication.

Your prompts and relevant reference images are sent to the selected image provider and to A2E's video service; the optional planning LLM also receives planning text. Gemini frames used for video are uploaded to A2E storage and made available through public CDN URLs that may expire. **`server/projects/<job_id>/` contains private prompts, planning results, saved images/videos, task IDs, event logs and provider result URLs.** Treat this folder and backups as sensitive; don't share or publish it. The folder is owner-only on POSIX and ignored by Git, but is not encrypted. Credentials are excluded from the saved project metadata. Job state survives backend restarts when this directory survives. Run a single backend worker; multiple workers or replicas sharing this directory are unsupported and can cause duplicate work.

## Configuration

The annotated template is `server/.env.example`. Nonempty A2E, Gemini and OpenAI keys saved in `.env` take precedence over old exported values after restart; unrelated settings use normal environment-first dotenv behavior. Running jobs retain the keys and model choices captured at acceptance. Never commit `.env`.

| Variable | Default | Meaning |
|---|---|---|
| `A2E_API_KEY` | empty | Required for every job, including Gemini-to-A2E. |
| `GEMINI_API_KEY` | empty | Required if Gemini is selected for text planning, images, or both. |
| `OPENAI_API_KEY` | empty | Required only for the explicit OpenAI GPT planning option; independent of `LLM_API_KEY`. |
| `A2E_BASE_URL` | `https://api.a2e.ai` | A2E API origin. |
| `A2E_IMAGE_MODEL` | `a2e` | Default A2E image ID when a programmatic job omits its choice. |
| `A2E_VIDEO_MODEL` | `a2e` | Default A2E video ID when a programmatic job omits its choice. |
| `A2E_AUTO_ADD_AUDIO` | `false` | Native A2E video audio switch; can affect cost. Other families have their own behavior. |
| `A2E_MAX_JOB_CREDITS` | `300` | Per-project sum of A2E generation quotes and the published R2 signing price, carried across resume attempts. This is **not** a cap on actual charges, Google/LLM charges, or other jobs. |
| `A2E_MAX_CONCURRENCY` | `2` | Maximum concurrently active A2E generations/uploads per job. |
| `A2E_POLL_INTERVAL_SECONDS` | `5` | Initial read-only polling delay. |
| `A2E_POLL_TIMEOUT_SECONDS` | `900` | Per-task timeout; timing out does not cancel the remote task. |
| `MAX_SCENES` | `2` | LLM-assisted scene limit, 1–4; simple planning uses one scene. |
| `MAX_SHOTS_PER_SCENE` | `3` | Limit, 1–5; simple planning uses at most three. |
| `MAX_CHARACTERS` | `2` | LLM-assisted visible-character limit, 0–5. |
| `LLM_API_KEY` | empty | Optional, separate text-planning API key. |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | Base URL of an OpenAI-compatible `/chat/completions` API. |
| `LLM_MODEL` | `gpt-4o-mini` | Model ID at that LLM provider. |
| `BACKEND_URL` | `http://127.0.0.1:8000` | Frontend's server-side proxy destination, set before `npm run build`. |

Limits reduce runaway work but do not guarantee a job completes within budget: prices are checked as the pipeline proceeds. A default simple job requests up to three frames and three clips; LLM-assisted jobs may also generate portraits and additional scenes. Quotes do not validate access, reserve a balance or guarantee final settlement. See [A2E's developer docs](https://video.a2e.ai/dev), Google's [Gemini text](https://aistudio.google.com/docs/text-generation) and [image](https://aistudio.google.com/docs/image-generation) documentation, and [OpenAI's models](https://developers.openai.com/api/docs/models) and [pricing](https://developers.openai.com/api/docs/pricing) for current availability and pricing.

## Integration and failure behavior

- A2E requests use the documented endpoint-specific body for the chosen family; a generation quote receives the same body and `/api/v1/...` route alias. The app checks HTTP status and A2E's JSON `code`, records task IDs, and polls the appropriate detail route. Failed/blocked tasks report known reason, trace and refund state.
- OpenAI GPT planning uses authenticated POSTs to `https://api.openai.com/v1/responses` with `store=false`. Story scenes, character descriptions and shots use strict structured-output schemas and are validated by the agents. Refusals, incomplete responses, invalid results and API errors stop the job rather than triggering a fallback. OpenAI may still retain data under its applicable API data policies; `store=false` does not override those policies.
- Gemini text and image generation send authenticated POSTs to `generativelanguage.googleapis.com/v1beta/interactions` with `store=false`. The text planner uses JSON schemas for scenes, characters and shots, validates results and stops on invalid output rather than silently falling back. Gemini, OpenAI GPT and the configured LLM are not covered by A2E's estimate guard. Verified Gemini image bytes are saved locally for references and later recovery. For a video frame, the app reads the current A2E `r2` feature price, reserves it against the estimate guard, requests `/v1/r2/upload-presigned-url`, then performs a raw-byte PUT using a **separate unauthenticated client**. The signed upload URL and bearer credentials are never written to the event log. The CDN URL is used only after PUT succeeds.
- **No paid creation POST, OpenAI/Gemini generation, presigned-URL request or storage PUT is automatically retried or silently switched to a fallback.** If a response is lost, generation or the upload-signing fee may already have been charged. Check OpenAI, Google and/or A2E task and billing records before trying again. Read-only task polling may retry transient errors with backoff and respects `Retry-After`.
- A job's locally recorded A2E quote and task events are in `server/projects/<job_id>/a2e_events.jsonl`; they exclude keys and signed upload URLs. `project.json` checkpoints the request, results, events, task IDs and cumulative estimates. Generated output URLs may expire; verified images and completed videos are downloaded promptly. Earlier requests may be billed even if another shot fails.

## Local API

- `GET /api/health` — configuration flags, without key values.
- `GET /api/models` — static OpenAI and Gemini text and image/video model catalog, default IDs, configured-LLM availability, clip durations and notes; not an access check.
- `POST /api/settings/a2e-key`, `POST /api/settings/gemini-key` and `POST /api/settings/openai-key` — save `{"api_key":"..."}`. All require JSON and `X-Requested-With: MicroDrama`; the response contains only configured/verified flags. This header is not authentication.
- `POST /api/generate` — accepts `idea`, `mode` (`idea2video` or `script2video`), optional `script`, `style`, `user_requirement`, `planning_provider` (`simple`, `gemini`, `openai` or `configured`), `planning_model` (selected provider's text model ID), `image_provider` (`a2e` or `gemini`), `image_model`, `video_model`, and integer `clip_duration`. In script mode, `script` is required. `planning_model` is only accepted for Gemini or OpenAI; omitting it selects that provider's catalog default. An explicitly selected provider requires its key and never silently falls back. For backward compatibility, omitting `planning_provider` uses the configured OpenAI-compatible LLM if present, otherwise simple planning. Omit `clip_duration` for the selected model's default, or use `null` for Veo's provider-determined duration. Omitted image/video choices retain the configured A2E defaults.
- `GET /api/status/{job_id}` — server-sent progress, quote, task, error and completion events; supports `Last-Event-ID` on reconnect.
- `GET /api/result/{job_id}` — durable result status, error and whether it can be resumed.
- `GET /api/projects` and `GET /api/projects/{job_id}` — saved project summaries, completed and uncertain steps, known A2E task IDs and links to available local assets. Provider keys are not returned; provider result URLs stay in private checkpoints.
- `POST /api/projects/{job_id}/resume` — manually resume a failed/interrupted project. Requires `X-Requested-With: MicroDrama` and `{"acknowledge_costs":true}`; add `"retry_uncertain":true` only after checking provider task/billing history and explicitly accepting the risk of another charge. These flags are consent, not authentication.
- `GET /api/projects/{job_id}/assets/{filename}` and restricted `GET /outputs/{job_id}/...` — local images and generated video files only. Project metadata and logs are not mounted as static files.

Example request (a real run may incur charges):

```bash
curl -X POST http://127.0.0.1:8000/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"idea":"A clockmaker discovers a door inside a clock", "mode":"idea2video", "planning_provider":"gemini", "planning_model":"gemini-3.8-flash", "image_provider":"a2e", "image_model":"a2e", "video_model":"a2e-v2-flash", "clip_duration":5}'
# For GPT planning instead (select your own accessible model):
curl -X POST http://127.0.0.1:8000/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"idea":"A clockmaker discovers a door inside a clock", "mode":"idea2video", "planning_provider":"openai", "planning_model":"gpt-6-luna", "image_provider":"a2e", "image_model":"a2e", "video_model":"a2e-v2-flash", "clip_duration":5}'
# Follow either request with GET /api/status/<returned-job-id>.
```

Programmatic clients are responsible for obtaining their own users' consent to actual provider charges.

## Tests

```bash
cd server
pip install -r requirements-dev.txt
python -m pytest -q
cd ../client
npm run lint
npm run build
```

Tests use mock transports and fake keys; they do not require provider credentials or spend credits. The test configuration points dotenv away from the real `server/.env` and uses temporary project directories. They cover model routing, GPT and Gemini planning, Gemini images, quote/submit bodies, private key storage, upload isolation, mocked end-to-end jobs, SSE replay and MoviePy concatenation. Recovery tests also simulate provider failures, process restarts, concurrent partial successes, saved Gemini references, lost submissions, in-flight storage faults, expired/missing assets, explicit retry consent and cumulative estimate limits. A real provider smoke test still requires your own keys, access and credits.

Adapted from [Anil-matcha/Open-AI-Micro-Drama-Generator](https://github.com/Anil-matcha/Open-AI-Micro-Drama-Generator).
