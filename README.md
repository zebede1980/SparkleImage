# ✨ SparkleImage

AI Photo Repair, Recovery & Enhancement Tool

SparkleImage is a containerized web application that connects to NanoGPT AI models to restore, colorize, enhance, and repair photographs. It runs on Linux ARM64 (Ampere) servers and provides an intuitive Gradio web interface.

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
   NANOGPT_MODEL=gpt-4o
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
  model: "gpt-4o"
  timeout: 120
  max_retries: 3
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
│   ├── clients/         # NanoGPT API client
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

## Face Preservation

SparkleImage strongly preserves facial features by:
1. Appending a strict face-preservation system prompt to every API request
2. Detecting faces before and after processing with OpenCV DNN
3. Warning the user if face count or detectability changes

## License

MIT
