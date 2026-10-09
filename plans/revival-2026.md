# SparkleImage — revival plan (2026-09-17)

Written after a full read of the existing code (9 commits, 29 files, ~2,200 lines) and a
comparison against the image pipeline that works in CardGenV2.

---

## 1. What the app was trying to achieve

From `plans/architecture.md`, the README and the code, the goal is a **self-hosted family-photo
restoration tool**:

> Upload an old, damaged or black-and-white photograph → pick an operation → get back the *same
> photograph*, repaired, with the people in it still recognisably themselves.

Concretely it promised seven operations — colorize, repair (creases/tears/stains), remove object
(masked), upscale 2×/4×, denoise, general enhance, remove scratches — plus batch processing, an
output gallery, an in-UI settings panel, EXIF preservation, and a face-preservation guarantee.
Delivery was a Gradio app in a Docker container on ARM64, port 7860.

The intended pipeline was:

```
upload → GPT-4o vision writes a better prompt → DALL-E does the edit → face check → save
```

That is a coherent product. The implementation of it is what went wrong.

---

## 2. Why it didn't work

Seven problems, roughly in order of severity. The first one is fatal on its own.

### 2.1 The generation model cannot do the job (fatal)

`Dalle3Client.edit_image()` calls `POST /images/edits`. That endpoint is **DALL-E 2 masked
inpainting**: it regenerates the *masked region* from a text prompt, with no fidelity to what was
underneath, and it ignores the unmasked pixels as anything other than context. DALL-E 3 has no
edit endpoint at all — which is why commit `b0874f1` downgraded the generation model to dall-e-2
and the following three commits (`21aa44b`, `467a423`, `c6eb307`) are all fighting masks and
multipart encoding.

So for whole-image operations (colorize, denoise, enhance, upscale, descratch, repair) the code
sends an almost-fully-transparent mask meaning "regenerate everything", and DALL-E 2 obligingly
paints a *new picture* loosely inspired by the prompt. There is no path in the architecture by
which "colorize *this* photo" could have produced the right answer. The model class was wrong,
not the wiring.

### 2.2 Geometry is destroyed on every call

`_make_square()` thumbnails the image to fit 1024×1024 and pads it with white. Every operation
therefore:

- caps output at 1024 px on the long edge (a scanned 4000 px photo comes back at a fraction of
  its resolution — and "Upscale 2×" returns something *smaller* than the input),
- returns a square with white bars,
- loses the original aspect ratio permanently, because nothing maps the result back.

`_resize_if_needed()` also pre-shrinks to `max_resolution` (2048) before that, so the configured
resolution setting does nothing.

### 2.3 Nothing composites the result back onto the original

Restoration is defined by "same photo, minus the damage". The code returns whatever the API
hands back as the whole output. Even with a good edit model, localized work (scratch removal,
object removal) should only change the pixels it needs to; everything else should be
bit-identical to the source. There is no mask-blend, no feather, no paste-back, no
resize-to-original step.

### 2.4 Face preservation never runs

`FaceDetector._load_model()` looks for `data/opencv_face_detector_uint8.pb` and
`data/opencv_face_detector.pbtxt`. Neither is in the repo, neither is in the Dockerfile, and
`MODEL_URL`/`CONFIG_URL` are defined but **never used by any code path** — nothing downloads
them. So `_load_model()` raises `FileNotFoundError`, `detect()` catches it, logs a warning and
returns `[]`. `has_faces()` is therefore always `False`, and `_process_image()` skips the
post-check entirely. The headline feature is inert.

Worse, even fully wired it only compares *face counts*. "Two faces before, two faces after" says
nothing about whether they are the same two people — which is the only question that matters.
The architecture doc specified embedding comparison; the code never got there.

### 2.5 The prompt-enhancement layer is aimed at the wrong kind of model

Every processor spends a GPT-4o vision call rewriting a short instruction into a long prose
DALL-E prompt, then prepends a ~60-word "You are an expert…" system prompt and sends the
concatenation as the edit prompt. Instruction-following image-to-image models (seedream,
flux-kontext, qwen-image) want the opposite: a short imperative — *"colourise this photograph
with natural 1950s skin tones"*. Long descriptive prompts push them toward regenerating the scene
rather than editing it. The vision call also doubles the latency and cost of every operation.

### 2.6 Seven processors are one processor

`colorize.py`, `denoise.py`, `enhance.py`, `repair.py`, `scratch.py`, `upscale.py` are
byte-identical apart from the prompt string and the status message — including a copy-pasted
`_enhance_prompt()` in each file. ~700 lines that should be one function and a table of
operations.

### 2.7 Assorted correctness and ops issues

- `time.sleep()` (blocking) inside the async retry loops in both clients — stalls the whole event
  loop during backoff, so one retrying request freezes every other user.
- `asyncio.get_event_loop().time()` used as the gallery filename timestamp; that's a *monotonic*
  clock, not wall time, so filenames are meaningless and restart-colliding.
- Batch processing is serial, and constructs + closes a fresh pair of HTTP clients per image.
- Dockerfile builds a wheel from `pyproject.toml`, installs it, **and then** copies `app/` in
  again — two copies of the code, the `COPY` one winning.
- Container runs as root (deliberately, per the last commit, to make bind mounts writable).
- No auth by default, on a box that is internet-facing via nginx proxy manager.
- No job/queue layer: an edit takes 30–120 s, and a phone locking mid-request loses the result
  (the exact failure we already fixed in CardGenV2).

---

## 3. What's needed to get there

The entire value of this app now sits in **one API call**: `source image + instruction → edited
image, subject preserved`. CardGenV2 already proved that call against nano-gpt. Everything else
is UX around it.

### 3.1 The core call (proven in CardGenV2 `src/scripts/api-image.js`)

```
POST {base}/images/generations
{
  "model": "<edit-capable model>",
  "prompt": "<short imperative instruction>",
  "image": "data:image/png;base64,…",     ← the source; this is the whole difference
  "n": 1,
  "response_format": "url",
  "seed": <random>,
  "width": …, "height": …, "size": "WxH"
}
```

Same OpenAI-compatible endpoint as text-to-image, with an `image` field. No multipart, no mask
gymnastics, no `/images/edits`. Multi-reference calls (e.g. "use this other photo of him as a
face reference") are a *different* endpoint — `POST {base}/images` with an `input_references`
array — as documented in CardGenV2's `/api/image/combine` route.

### 3.2 Model handling to carry over verbatim

Three hard-won details from CardGenV2 that will otherwise bite again:

1. **Seedream has its own resolution enum and a ~3.7 MP minimum.** Send flux-style 1024×1024
   dims and it *silently returns a square at the wrong size* instead of erroring. The working
   map (from nano-gpt's `seedream-v5.0-lite` `supported_parameters.resolutions`):
   1:1 → 2048×2048, 9:16 → 1440×2560, 16:9 → 2560×1440, 3:4 → 2048×3072, 4:3 → 3072×2048.
   Note this also solves §2.2 — seedream natively outputs 2–6 MP, so resolution stops being a
   compromise.
2. **Silent safety blocks.** flux-kontext returns a solid black image with HTTP 200 when its
   filter rejects the source. CardGenV2 detects this by sampling a 32×32 downscale and checking
   luminance variance < 4, then raises a real error naming the cause. Photos of real people trip
   these filters often enough that this is mandatory, not optional.
3. **Per-model capability marks.** Not every model can edit; the ones that can aren't reliably
   named. CardGenV2 keeps user-tickable `generate / edit / combine / upscale` marks per model
   with a conservative name-based guess as the default. Same registry here, so every dropdown is
   filtered to models that can actually do the operation.

### 3.3 Restore geometry discipline

- Never pad to square. Choose the supported resolution closest to the source's aspect ratio.
- Send the source at full resolution (subject to the model's input limits), not pre-shrunk.
- Resize the returned image back to the original pixel dimensions before saving.
- For localized operations (object removal, scratch removal), composite: feather the mask and
  blend the edit back *only* inside it, leaving the rest of the photograph untouched.
- Keep the original file alongside every output; never overwrite.

### 3.4 Operations become data, not classes

Delete `app/processors/*` and replace with one table:

```python
OPERATIONS = {
  "colorize": Op(label="Colorize", instruction="Colourise this photograph…",
                 params=["era_hint"], needs_mask=False, composite=False,
                 capability="edit"),
  "scratch":  Op(label="Remove scratches", instruction="Remove the scratches and dust specks…",
                 params=["severity"], needs_mask=False, composite=True,
                 capability="edit"),
  …
}
```

One `run_operation(image, op_id, params, mask=None)` executes all of them. Adding an operation
becomes a four-line entry.

### 3.5 Face preservation, done honestly

Three tiers; pick per the decision in §5:

- **Minimum (always):** a before/after comparison slider in the UI, and the original kept on
  disk. The human judges. This is the only *guarantee* that is actually true.
- **Real check:** ship ArcFace/InsightFace embeddings in the image, compute cosine similarity
  between the pre- and post-edit face crops, and surface the number ("identity similarity 0.91").
  Advisory, not a gate — but an honest one, and it needs the model weights baked into the
  Dockerfile rather than assumed present.
- **Actually improves results:** detect the face, crop it with padding, run the edit on the crop
  at full resolution, paste it back. Small-in-frame faces are where generative edits destroy
  identity; giving the model the face at high resolution is the single biggest quality lever.

### 3.6 Jobs and resumability

Edits run 30–120 s and this will be used from a phone. Port the CardGenV2 pattern: a server-side
job registry keyed by client ref, buffered results persisted to disk, and a poll/collect route,
so a locked phone rejoins instead of losing a paid-for result. (Directly analogous to
`cardgenv2_image_resumable_jobs`.)

### 3.7 Upscale needs a real upscaler

Asking a generative edit model to "upscale 2×" gets you a reimagining at that model's native
size. Use either a model marked with the `upscale` capability (nano-gpt's enhance/upscale
family, as CardGenV2's Image Enhance path does) or a local Real-ESRGAN pass in the container.
Either way it is a different code path from the instruction edits.

### 3.8 The feature this unlocks — one-click Restore

The architecture doc listed a chained pipeline as "suggested extra #8". With a real edit model
it becomes *the* headline feature: **descratch → repair → denoise → colorize → upscale**, each
step feeding the next, with an optional review-and-accept between steps. That is the thing a
person with a shoebox of damaged photos actually wants, and it was unreachable with DALL-E.

### 3.9 Housekeeping

Fix the blocking `time.sleep` in async retry, use wall-clock timestamps for filenames, drop the
duplicate code copy from the Dockerfile, run as non-root with a host-UID mapping, turn on auth
before it goes anywhere near the proxy, and make batch concurrent with a bounded worker pool.

---

## 4. What survives the rewrite

| Component | Verdict |
| --- | --- |
| `app/config/` (pydantic + YAML + env override) | **Keep.** Clean, works, good shape. |
| `app/ui/app.py` Gradio shell, tabs, gallery, settings | **Keep the layout**, rewire the internals. `gr.ImageEditor` gives the mask brush free. |
| Docker / compose / ARM64 build | **Keep**, with the fixes in §3.9. |
| `app/clients/dalle3.py` | **Delete.** Wrong endpoint, wrong model class. |
| `app/clients/nanogpt.py` | **Keep ~30%** — the async client + retry skeleton. Retarget it. |
| `app/processors/*` (7 files) | **Delete.** Replaced by the operations table (§3.4). |
| `app/utils/prompts.py` | **Rewrite.** Long prose prompts → short imperatives. |
| `app/utils/face.py` | **Rewrite.** Ship the weights, do embeddings, or drop the claim. |
| `plans/architecture.md` | **Keep as history**; this document supersedes it. |

Net: roughly 1,100 of the 2,200 lines go, and the replacement is smaller than what it replaces.

---

## 5. Decisions taken (2026-09-17)

1. **UI stack — keep Gradio.** Rewire the existing tabs rather than rebuild. `gr.ImageEditor`
   gives the mask brush free, and the fastest route to judging real output on a real photo wins.
   Revisit a CardGenV2-style frontend only if the results justify it.
2. **Face preservation — crop, edit at full resolution, paste back** (§3.5, third tier), plus the
   before/after slider. The automated count-check is deleted, not repaired.
3. **Model catalogue — fetched live**, see §6. Supersedes the guesswork in §3.2.

---

## 6. Confirmed model catalogue (fetched 2026-09-17)

Retrieved with the nano-gpt key from CardGenV2's config.

### 6.1 The catalogue endpoint is not the one CardGenV2 uses

`GET /api/v1/models` — what `api.js:fetchModels()` calls — returns **595 text models and
essentially no image models** (`z-image-turbo`, the user's own configured image model, is not in
it). The real catalogue is:

```
GET https://nano-gpt.com/api/models
→ { "models": { "text": {...}, "image": {...214...}, "video": {...}, "3d": {...} } }
```

and every image model carries structured metadata:

| Field | Meaning | Coverage |
| --- | --- | --- |
| `iconLabel` | `text-to-image` / `image-to-image` / `both` | 199/214 |
| `supportsMultipleImg2Img` | accepts several reference images in one call | 89 |
| `censored` | whether a content filter is applied | 53 |
| `resolutions` | the exact output sizes the model accepts | 214 |
| `additionalParams.customResolution` | free-form W×H, with min/max/step/minTotalPixels | some |
| `inputImageConstraints` | e.g. `maxBytes: 10485760` | 12 |
| `cost` | per-resolution price | 214 |
| `maxImages` | outputs per call | 214 |

**This means capability marks can be derived from the API rather than guessed from model names.**
CardGenV2's `guessImageModelCapabilities()` name-marker heuristic exists only because it was
reading the wrong endpoint — `iconLabel` and `supportsMultipleImg2Img` answer `edit` and
`combine` authoritatively. SparkleImage should read `/api/models` from the start, and this is
worth back-porting to CardGenV2.

### 6.2 Core edit model: `seedream-v4.5`

| | |
| --- | --- |
| `iconLabel` | `both` — text-to-image *and* image-to-image |
| `censored` | **false** |
| Max output | 4096×4096 (16.8 MP); 11 preset resolutions from 1920×1920 up |
| `customResolution` | **enabled** — 1024–4096, step 64, `minTotalPixels: 3686400` |
| `supportsMultipleImg2Img` | true |
| Input limit | 10 MB per image |
| Cost | $0.04/image at any resolution |

Three of those matter a great deal here:

- **`censored: false`** — photographs of real people, especially old family photos, trip content
  filters constantly. This is the single most important attribute for this app, and it is why
  `seedream-v5.0-lite` (censored, 9.4 MP, $0.035) is the *second* choice despite being cheaper.
- **`customResolution` with step 64** — supersedes §3.2's fixed ratio map entirely. Rather than
  snapping to the nearest preset aspect ratio, compute the source photo's own ratio, round to a
  64-px grid within 1024–4096, and scale up until total pixels ≥ 3,686,400. The output then
  matches the original's framing exactly, which is precisely what §2.2/§3.3 needed. The 3.69 MP
  floor is real and silent: go under it and the model quietly returns a square.
- **16.8 MP ceiling** — a 4000 px scan can come back at full size. Under DALL-E this app was
  capped at 1024.

`seedream-v4.5-sequential` (same specs, `maxImages: 15`) is worth keeping in the list: it can
return several restoration variants from one call, which suits "show me three colourisations,
pick one".

### 6.3 Shortlist to mark edit-capable

All `censored` false-or-unset, all `supportsMultipleImg2Img`:

- `flux-2-pro-image-to-image`, `flux-2-max-image-to-image` — CardGenV2's defaults, known-good
- `nano-banana-pro-edit`, `nano-banana-pro-edit-ultra` — strong instruction following
- `qwen-image-max-edit`, `qwen-image-3-pro`
- `glm-image-edit` — also has `customResolution`
- `wan-2.6-image-edit`, `gemini-flash-edit`

Avoid for this app: `openai/gpt-image-2.5/*/edit` and `reve/2.1/*` (all `censored: true`).

### 6.4 Upscaling is a solved problem, with a real model

§3.7 no longer needs Real-ESRGAN in the container. `clarity-ai-pro-upscaler` is explicitly built
for this job — *"photorealistic detail, identity preservation, predictable target-megapixel
output"* — and exposes exactly the right controls:

- `target_megapixels`: 1–64, default 4
- `creativity`: −10 to +10, **default 0** — at or below zero it sharpens what is there instead
  of inventing detail, which is the correct setting for restoring a real person's photograph

Cost $0.12/image. Cheaper alternatives: `seedvr2-image` (2K/4K/8K, $0.01), `pruna-ai/p-image/upscale`
(1–8 MP target, $0.005), `Upscaler` (plain 2×/4×, $0.005). All are `image-to-image`.

### 6.5 Consequences for the plan

- §3.2's hand-maintained seedream ratio map is replaced by `customResolution` arithmetic.
- §3.2's name-guessed capability marks are replaced by `iconLabel` / `supportsMultipleImg2Img`.
- §3.7's local upscaler is replaced by `clarity-ai-pro-upscaler` at `creativity: 0`.
- The blank-image safety-block detector (§3.2 item 2) stays — it is cheap insurance — but
  defaulting to `censored: false` models should make it fire rarely, and the UI should warn when
  a censored model is selected for a photo containing people.
