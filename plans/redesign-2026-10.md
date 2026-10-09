# SparkleImage — redesign (2026-10-09)

Follows `revival-2026.md`. That plan fixed the plumbing (right endpoint, geometry, catalogue,
async client). This one changes the *restoration method*, based on measured results rather than
guesses: a day of experiments on real family photos — a phone shot of a faded 1980s colour slide,
1940s black-and-white flatbed scans with creases and foxing, and a 333×537 thumbnail.

---

## 1. What the experiments showed

Identity was measured with ArcFace (insightface `w600k_r50`): cosine similarity between each face
in the source and the same face in the result. For reference, a SeedVR2-only upscale — which
invents nothing — scores ~0.93 on the slide photo; that is the ceiling.

| Finding | Evidence | Consequence |
|---|---|---|
| One instruction-following edit does the whole restoration well | Qwen-Image 2.1 Edit (local, ~25 s at 1.5 MP) removed the cast, sharpened, repaired a crease and colourised, composition intact | **One generative pass, not a chain.** `RESTORE_CHAIN` (descratch → repair → denoise → colourise → upscale) compounds drift five times |
| The seed matters as much as the model | Same model and prompt, six seeds: the child's face scored 0.72–0.82 | **Generate several candidates and rank them** |
| ArcFace ranking matches what a person sees | Every ranking agreed with side-by-side visual review | Automatic per-face selection is trustworthy |
| Best face per person, composited from different seeds, is seamless | Lab mean/std match + feathered ellipse; no visible seam at 2× | **Per-face compositing across candidates** |
| Re-restoring zoomed face crops makes identity *worse* | 12/12 trials scored lower than the whole-frame result | **Remove `refine_faces` / `face_crop_edit`** — `revival`'s headline mechanism is backwards |
| ArcFace can't choose the base image | Seedream 4.5 scored highest on every face but restored least (kept screw heads, softer, warmer) — it scores high partly by changing less | **Base image = human choice** (default to best local seed); ArcFace picks faces only |
| Pre-white-balancing doesn't help | WB'd input averaged the same as raw | Don't add a prep stage for colour |
| SeedVR2 7B is the right finishing upscaler | Faithful, ~30–40 s for 2×; alone it can't fix a very soft source | Finish with it; never use it as the restoration |
| Good scans lose detail through any generative model | A 16 MP scan comes back at ~2 MP of model output | **Faithful mode:** keep the scan's own luminance at full resolution, take only colour from the AI |
| Cloud models are not better than local | Seedream 4.5, FLUX.2 max ≈ local seeds; Seedream 5 Pro, Nano Banana Pro, Qwen 2.1 Pro worse; GPT Image 2 failed | Cloud = optional extra candidates, a few pence each |
| Seedream 4.5 / Qwen 2.1 Pro return 2048² unless sized | Observed | Keep `revival`'s geometry code |

## 2. The new pipeline

```
source ─┬─ prep: EXIF orientation, detect B&W vs colour → pick instruction
        │
        ├─ candidates: N × local Qwen-Image 2.1 Edit (seeds)          ← engine: ComfyUI on the home PC
        │              + optional nano-gpt models (Seedream 4.5, FLUX.2 max)
        │
        ├─ identity: detect faces in source (YuNet), embed (ArcFace), score every face in every candidate
        │
        ├─ base: best-scoring local candidate by default; user can pick any candidate
        ├─ faces: per person, best-scoring version from the whole pool → Lab-matched feathered paste
        │         (user can override per face)
        │
        └─ finish, two outputs:
             restored  = composite → SeedVR2 7B → up to the configured size
             faithful  = scan's luminance at full res + composite's colour (Lab a/b)   [CPU, free]
```

Both finishes are produced for every job, because the right one depends on the photo: a soft or
damaged source wants **restored**; a sharp scan you want to keep honest wants **faithful**
(which keeps creases and paper texture — a masked local repair can fix those later).

## 3. Architecture

Keep: Gradio, `pydantic-settings` config, `clients/catalog.py`, `clients/image_api.py`,
`core/geometry.py`, `core/gallery.py`, blank-frame detection, YuNet bootstrap in the Dockerfile.

Cut: `core/pipeline.refine_faces`, `ProcessingConfig.face_crop_edit/face_padding`,
`operations.RESTORE_CHAIN` and the separate descratch / denoise / enhance operations (one
instruction covers them), the Batch tab (multi-upload replaces it).

New modules:

| Module | Role |
|---|---|
| `clients/comfy.py` | Async ComfyUI client: upload, queue, poll history, fetch. Talks to the home PC through the existing `comfy-gateway` (X-API-Key). Tolerates ComfyUI's multi-second stalls during model swaps, as CardGenV2 learned the hard way |
| `clients/workflows.py` | API-format graphs: Qwen-Image 2.1 edit, SeedVR2 upscale |
| `core/engines.py` | One interface over both back ends: `edit(image, instruction, seed)` and `upscale(image, short_edge)` |
| `core/identity.py` | YuNet detection + ArcFace embedding via onnxruntime (no `insightface` package — it needs a compiler and has no ARM64 wheels); face matching across candidates |
| `core/composite.py` | Lab colour match, feathered face paste, chroma transfer |
| `core/restore.py` | The pipeline above as plain functions over images — no UI, no I/O; unit-testable |
| `core/jobs.py` | Persistent job store (`data/jobs/<id>/job.json` + images) and a single-worker queue — one GPU, so one job at a time. Work runs independently of the browser: closing the tab or a phone locking loses nothing |
| `ui/app.py` | Rewritten: Restore (multi-upload + options) → Jobs (live status, candidate grid with scores, base picker, per-face overrides, instant re-composite) → Gallery → Settings |
| `scripts/restore_folder.py` | Headless CLI over a folder — the test bench |

Engines run on the home PC GPU (RTX 4080 Super 16 GB); identity and compositing run in the
container on the OCI ARM box (CPU, well under a second per face).

## 4. Config additions

```yaml
comfy:
  url: https://comfy.terminus.giize.com   # same gateway CardGenV2 uses
  api_key: ...                            # gateway X-API-Key
  edit_megapixels: 1.5
  upscale_short_edge: 2160                # SeedVR2 target; 16 GB is comfortable here
restore:
  local_seeds: 4
  cloud_models: []                        # e.g. ["seedream-v4.5", "flux-2-max-image-to-image"]
  face_swap_margin: 0.02                  # only swap a face in if it beats the base by this much
```

Model files needed on the PC (all already present): `qwen_image_2.1_int8_convrot`,
`qwen3vl_8b_int8_convrot`, `qwen_image_2.1_vae_bf16`, `seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16`,
`ema_vae_fp16`. ArcFace `w600k_r50.onnx` (166 MB, insightface — **non-commercial licence**, fine
for family photos) is fetched at image build time like YuNet.

## 5. Milestones

1. **Core** — comfy client, workflows, identity, composite, restore pipeline, folder CLI.
   Run every sample; contact sheet. Tests for identity matching, compositing, chroma transfer,
   workflow graphs, comfy client (mock transport).
2. **Jobs** — store, worker, restart recovery (in-flight jobs → failed with a clear message).
3. **UI** — rewrite around jobs.
4. **Ship** — Dockerfile (ArcFace download, onnxruntime), compose, `.env.example`, README;
   deploy to OCI behind NPM with auth. *Needs Joe: NPM host + auth credentials.*

Later: masked local repair (faithful mode + brush over a crease → Qwen inpaint of just that area),
auto-crop of scan borders, DDColor as a deterministic colouriser option.
