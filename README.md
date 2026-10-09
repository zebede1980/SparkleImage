# ✨ SparkleImage

Old photographs repaired, colourised and sharpened — with the people in them
still themselves.

Upload a faded, damaged or black-and-white photo. SparkleImage makes several
independent restorations of it, measures how well each one keeps every
person's face, builds the result from the best version of each face, and
upscales it. You get two finished versions:

- **Restored** — fully repaired and sharpened, upscaled with SeedVR2.
- **Faithful** — the scan's own detail at its full resolution, with colour
  from the restoration. Nothing invented, but damage in the scan stays.

## How it works

```
photo ─┬─ N restorations (Qwen-Image 2.1 Edit on your GPU; optional cloud models)
       ├─ every face in every restoration scored for likeness (ArcFace)
       ├─ best restoration as the base + best version of each face pasted in
       └─ SeedVR2 upscale  →  restored.png
          scan luminance + restored colour  →  faithful.jpg
```

Why this shape, with the measurements behind each decision, is in
[`plans/redesign-2026-10.md`](plans/redesign-2026-10.md). The short version:

- **One edit does the whole job.** A single instruction ("restore and colourise,
  keep every person the same") beats a chain of descratch → repair → denoise →
  colourise, which compounds drift at every step.
- **The seed matters as much as the model.** The same model and instruction,
  six seeds: one child's likeness ranged 0.72–0.82. So several are made.
- **ArcFace ranks faces the way a person does**, so the best version of each
  face is picked automatically — you can override any pick.
- **Re-editing zoomed face crops makes faces worse** (12 of 12 trials), so that
  isn't done.
- **Faces with too little detail in the original** (tiny, grainy) can't be
  scored reliably; those are flagged for you to check by eye.

Jobs run in the background, one at a time, and everything is saved to
`data/jobs/`. Closing the page or locking your phone loses nothing.

## What runs where

| Part | Where |
| --- | --- |
| UI, job queue, face scoring, compositing | This container (CPU) |
| Restorations (Qwen-Image 2.1 Edit) and upscaling (SeedVR2 7B) | ComfyUI on a GPU PC, through `comfy-gateway` |
| Optional extra candidates, Mark & remove | nano-gpt (`seedream-v4.5`, `flux-2-max-image-to-image`, …) |

ComfyUI needs these files in its models folders (diffusion_models, text_encoders, vae, SEEDVR2):
`qwen_image_2.1_int8_convrot.safetensors`, `qwen3vl_8b_int8_convrot.safetensors`,
`qwen_image_2.1_vae_bf16.safetensors`, `seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors`,
`ema_vae_fp16.safetensors`, and the SeedVR2 custom node. On a 16GB card a
restoration takes ~25 s and an upscale ~40 s.

## Quick start

### Docker

1. `cp .env.example .env`; set `COMFY_URL`, `COMFY_API_KEY`, and **both**
   `AUTH_USERNAME` and `AUTH_PASSWORD`.
2. `docker compose up --build -d` (joins the external `nginx-proxy-nw` network;
   the port is bound to localhost only — publish it through Nginx Proxy Manager).

The image bakes in the face models (YuNet, and ArcFace `w600k_r50` from
insightface — **licensed for non-commercial use only**) at `/app/models`.

### Local

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
mkdir -p data/models
curl -L -o data/models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
curl -L -o /tmp/buffalo_l.zip https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip \
  && unzip -j /tmp/buffalo_l.zip w600k_r50.onnx -d data/models
SPARKLE_COMFY__URL=http://127.0.0.1:8188 python -m app.main
```

Without the ArcFace model the app still restores, but can't compare faces; it
says so at startup.

### Headless

```bash
python scripts/restore_folder.py photos/ out/ --comfy http://127.0.0.1:8188 --seeds 4
```

Writes every candidate, the composite, both finishes and a `report.json` of
scores per photo.

## Configuration

Settings live in `data/config.yaml`, editable from the Settings tab. Any value
can be set with a `SPARKLE_`-prefixed environment variable using `__` for
nesting (`SPARKLE_COMFY__URL`). Non-empty environment variables take precedence
over the saved file.

```yaml
comfy:
  url: ""                  # comfy-gateway URL
  api_key: ""              # the gateway's X-API-Key
  edit_megapixels: 1.5     # above ~2 spills a 16GB card's VRAM
  upscale_short_edge: 2160
restore:
  local_seeds: 4
  cloud_models: []         # e.g. [seedream-v4.5]
  face_swap_margin: 0.02
image:                     # nano-gpt
  base_url: "https://nano-gpt.com/api/v1"
  api_key: ""
  edit_model: "seedream-v4.5"   # Mark & remove
ui:
  auth_username: null
  auth_password: null
```

## Project structure

```
app/
├── clients/
│   ├── comfy.py       # ComfyUI over the gateway: upload, queue, poll, fetch
│   ├── workflows.py   # Qwen-Image 2.1 edit and SeedVR2 graphs
│   ├── catalog.py     # nano-gpt model catalogue and capabilities
│   └── image_api.py   # nano-gpt generate / edit / upscale, blank-frame detection
├── core/
│   ├── restore.py     # the pipeline: candidates → scores → composite → finishes
│   ├── jobs.py        # persistent jobs and the single background worker
│   ├── engines.py     # one interface over local and cloud models
│   ├── identity.py    # ArcFace likeness scoring
│   ├── faces.py       # multi-scale YuNet face detection with landmarks
│   ├── composite.py   # colour matching, face pasting, chroma transfer
│   ├── geometry.py    # output sizing for cloud models
│   ├── pipeline.py / operations.py   # Mark & remove
│   └── runtime.py     # config, catalogue cache, worker wiring
├── config/            # pydantic settings + YAML persistence
├── ui/                # Gradio app and the job views
└── main.py
scripts/restore_folder.py   # headless batch runner
```

## Tests

```bash
pip install -e ".[dev]" && pytest
```

## History

- **0.1** used DALL-E's `/images/edits` (masked inpainting): restoration was
  unreachable by construction. Post-mortem: `plans/revival-2026.md`.
- **revival (2026-09)** moved to nano-gpt image-to-image models with proper
  geometry handling.
- **redesign (2026-10)** moved restoration to a local GPU with multiple
  candidates and per-face likeness selection: `plans/redesign-2026-10.md`.

## License

MIT. The ArcFace model weights the image downloads are not MIT: insightface
licenses them for non-commercial use only.
