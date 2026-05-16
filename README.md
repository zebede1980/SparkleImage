# ✨ SparkleImage

AI Photo Repair, Recovery & Enhancement Tool

SparkleImage is a containerized web application that uses a **multi-model AI architecture** to restore, colorize, enhance, and repair photographs. It combines **GPT-4o** for vision analysis and prompt enhancement with **DALL-E 3** for high-quality image generation and editing. Runs on Linux ARM64 (Ampere) servers with an intuitive Gradio web interface.

## Architecture

```
User Upload → GPT-4o (Vision Analysis) → Enhanced Prompt → DALL-E 3 (Image Generation) → Face Validation → Output
```

- **Vision Model (GPT-4o)**: Analyzes images, detects damage, and crafts precise editing prompts
- **Generation Model (DALL-E 3)**: Performs the actual image editing, colorization, and restoration
- **Face Preservation**: OpenCV DNN validation ensures people remain recognizable

## Features

- **Colorize** — Convert black & white or sepia photos to realistic color
- **Repair** — Restore damaged photos (creases, tears, stains)
- **Remove Object** — Highlight and remove unwanted objects with inpainting
- **Upscale** — Increase resolution 2× or 4× with detail enhancement
- **Denoise** — Remove grain, scanner noise, and compression artifacts
- **Enhance** — General clarity, contrast, and color balance improvement
- **Remove Scratches** — Dedicated thin-line scratch and dust removal
- **Face Preservation** — Strong prompting and validation to keep people recognizable
- **Batch Processing** — Apply operations to multiple images at once
- **Gallery** — View and download recent outputs

## Quick Start

### Using Docker Compose (Recommended)

1. Clone the repository and navigate to the project directory.
2. Create a `.env` file with your NanoGPT credentials:
   ```bash
   NANOGPT_API_KEY=your_api_key_here
   NANOGPT_API_URL=https://api.nanogpt.com/v1
   NANOGPT_VISION_MODEL=gpt-4o
   NANOGPT_GENERATION_MODEL=dall-e-3
   USE_PROMPT_ENHANCEMENT=true
   ```
3. Build and run:
   ```bash
   docker compose up --build -d
   ```
4. Open your browser to `http://localhost:7860`.

### Local Development

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -e .
   ```
3. Run the application:
   ```bash
   python -m app.main
   ```

## Configuration

Settings are stored in `data/config.yaml` and can be edited via the **Settings** tab in the UI. You can also override values using environment variables with the `SPARKLE_` prefix (e.g., `SPARKLE_NANOGPT__API_KEY`).

### Example `data/config.yaml`

```yaml
nanogpt:
  api_url: "https://api.nanogpt.com/v1"
  api_key: ""
  timeout: 120
  max_retries: 3
  vision_model:
    model: "gpt-4o"
    enabled: true
  generation_model:
    model: "dall-e-3"
    enabled: true
  use_prompt_enhancement: true
processing:
  max_resolution: 2048
  output_format: "png"
  preserve_exif: true
  face_preservation: true
  default_quality: "high"
  jpeg_quality: 95
ui:
  server_name: "0.0.0.0"
  server_port: 7860
  share: false
  auth_username: null
  auth_password: null
  theme: "default"
```

## Project Structure

```
SparkleImage/
├── app/
│   ├── config/          # Configuration models and YAML manager
│   ├── clients/         # NanoGPT & DALL-E 3 API clients
│   ├── processors/      # Image processing pipelines
│   ├── utils/           # Face detection, prompts, helpers
│   ├── ui/              # Gradio web interface
│   └── main.py          # Application entry point
├── data/
│   ├── config.yaml      # Saved settings
│   ├── uploads/         # Temporary uploads
│   └── output/          # Processed images
├── Dockerfile           # Multi-stage ARM64 build
├── docker-compose.yml   # Container orchestration
├── pyproject.toml       # Python project metadata
└── README.md
```

## Multi-Model Pipeline

Each processing operation follows this flow:

1. **Image Upload** — User uploads photo via Gradio interface
2. **Vision Analysis** (Optional) — GPT-4o analyzes the image and enhances the editing prompt for better results
3. **Image Generation** — DALL-E 3 receives the enhanced prompt and original image, performs the edit
4. **Face Validation** — OpenCV DNN detects faces before/after, warns if identity is compromised
5. **Output** — Result saved to gallery with EXIF preservation (if enabled)

## Face Preservation

SparkleImage strongly preserves facial features by:
1. Appending a strict face-preservation system prompt to every API request
2. Detecting faces before and after processing with OpenCV DNN
3. Warning the user if face count or detectability changes

## Model Recommendations

| Task | Recommended Model | Notes |
|------|-------------------|-------|
| Vision Analysis | GPT-4o | Best image understanding and prompt crafting |
| Image Generation | DALL-E 3 | High-quality edits, colorization, inpainting |
| Budget Option | GPT-4o-mini | Adequate for simple prompt enhancement |

## License

MIT
