"""Gradio web UI for SparkleImage."""

import asyncio
import logging
from pathlib import Path
from typing import Optional

import gradio as gr
from PIL import Image

from app.clients.dalle3 import Dalle3Client
from app.clients.nanogpt import NanoGPTClient
from app.config.manager import get_config, get_config_manager
from app.config.models import AppConfig
from app.processors import (
    ColorizeProcessor,
    DenoiseProcessor,
    EnhanceProcessor,
    InpaintProcessor,
    RepairProcessor,
    ScratchProcessor,
    UpscaleProcessor,
)
from app.processors.base import ProcessingResult
from app.utils.face import FaceDetector

logger = logging.getLogger(__name__)

# Registry of available processors
PROCESSORS = {
    "colorize": ColorizeProcessor,
    "repair": RepairProcessor,
    "remove_object": InpaintProcessor,
    "upscale": UpscaleProcessor,
    "denoise": DenoiseProcessor,
    "enhance": EnhanceProcessor,
    "scratch": ScratchProcessor,
}


def _get_clients() -> tuple[NanoGPTClient, Dalle3Client]:
    """Create vision and generation clients from current config."""
    config = get_config()
    vision_client = NanoGPTClient(config.nanogpt)
    generation_client = Dalle3Client(config.nanogpt)
    return vision_client, generation_client


async def _process_image(
    operation: str,
    image: Image.Image,
    mask: Optional[Image.Image],
    **kwargs: object,
) -> tuple[Image.Image, str]:
    """Run a processing operation and return result with status message."""
    if image is None:
        return Image.new("RGB", (1, 1)), "No image provided."

    config = get_config()
    vision_client, generation_client = _get_clients()
    processor_cls = PROCESSORS.get(operation)
    if processor_cls is None:
        return image, f"Unknown operation: {operation}"

    processor = processor_cls(vision_client, generation_client, config.processing)
    face_detector = FaceDetector()

    # Pre-check faces
    has_faces = face_detector.has_faces(image)
    face_msg = "Faces detected." if has_faces else "No faces detected."

    try:
        result: ProcessingResult = await processor.process(image, mask=mask, **kwargs)
    except Exception as exc:
        logger.exception("Processing failed")
        return image, f"Error: {exc}"
    finally:
        await vision_client.close()
        await generation_client.close()

    if not result.success:
        return image, result.message

    # Post-check face preservation
    if has_faces and config.processing.face_preservation:
        passed, check_msg = face_detector.compare_face_presence(image, result.image)
        if not passed:
            return result.image, f"{result.message} | {check_msg}"
        face_msg = check_msg

    # Save to output gallery
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = asyncio.get_event_loop().time()
    out_path = output_dir / f"{operation}_{int(timestamp)}.png"
    result.image.save(out_path)

    return result.image, f"{result.message} | {face_msg} | Saved to {out_path.name}"


def create_app() -> gr.Blocks:
    """Build and return the Gradio application."""
    config = get_config()

    with gr.Blocks(title="SparkleImage - AI Photo Repair & Enhancement") as app:
        gr.Markdown("# ✨ SparkleImage")
        gr.Markdown(
            "AI-powered photo repair, colorization, enhancement, and restoration. "
            "All operations strongly preserve facial features."
        )

        with gr.Tabs():
            # --- Single Image Tab ---
            with gr.TabItem("Single Image"):
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(
                            label="Upload Image",
                            type="pil",
                            sources=["upload", "clipboard"],
                        )
                        operation = gr.Dropdown(
                            choices=[
                                ("Colorize", "colorize"),
                                ("Repair Damage", "repair"),
                                ("Remove Object", "remove_object"),
                                ("Upscale", "upscale"),
                                ("Denoise", "denoise"),
                                ("Enhance", "enhance"),
                                ("Remove Scratches", "scratch"),
                            ],
                            value="enhance",
                            label="Operation",
                        )

                        # Dynamic parameters per operation
                        with gr.Group() as params_group:
                            era_hint = gr.Textbox(
                                label="Era Hint (for Colorize)",
                                placeholder="e.g., 1950s",
                                visible=False,
                            )
                            damage_type = gr.Textbox(
                                label="Damage Type (for Repair)",
                                value="creases and tears",
                                visible=False,
                            )
                            object_desc = gr.Textbox(
                                label="Object to Remove (for Remove Object)",
                                placeholder="e.g., the red car",
                                visible=False,
                            )
                            scale = gr.Radio(
                                choices=[2, 4],
                                value=2,
                                label="Scale (for Upscale)",
                                visible=False,
                            )
                            noise_type = gr.Textbox(
                                label="Noise Type (for Denoise)",
                                value="grain and noise",
                                visible=False,
                            )
                            strength = gr.Dropdown(
                                choices=["low", "medium", "high"],
                                value="medium",
                                label="Strength (for Denoise)",
                                visible=False,
                            )
                            focus = gr.Textbox(
                                label="Focus (for Enhance)",
                                value="clarity and color",
                                visible=False,
                            )
                            severity = gr.Dropdown(
                                choices=["light", "moderate", "heavy"],
                                value="moderate",
                                label="Severity (for Scratch)",
                                visible=False,
                            )

                        process_btn = gr.Button("Process Image", variant="primary")

                    with gr.Column():
                        output_image = gr.Image(label="Result", type="pil")
                        status_text = gr.Textbox(label="Status", interactive=False)

                # Show/hide parameters based on operation
                def update_params(op: str) -> dict:
                    return {
                        era_hint: gr.update(visible=op == "colorize"),
                        damage_type: gr.update(visible=op == "repair"),
                        object_desc: gr.update(visible=op == "remove_object"),
                        scale: gr.update(visible=op == "upscale"),
                        noise_type: gr.update(visible=op == "denoise"),
                        strength: gr.update(visible=op == "denoise"),
                        focus: gr.update(visible=op == "enhance"),
                        severity: gr.update(visible=op == "scratch"),
                    }

                operation.change(
                    update_params,
                    inputs=operation,
                    outputs=[
                        era_hint,
                        damage_type,
                        object_desc,
                        scale,
                        noise_type,
                        strength,
                        focus,
                        severity,
                    ],
                )

                async def on_process(
                    img, op, era, damage, obj_desc, scl, ntype, stren, foc, sev
                ):
                    kwargs = {}
                    if op == "colorize":
                        kwargs["era_hint"] = era
                    elif op == "repair":
                        kwargs["damage_type"] = damage
                    elif op == "remove_object":
                        kwargs["object_description"] = obj_desc
                    elif op == "upscale":
                        kwargs["scale"] = scl
                    elif op == "denoise":
                        kwargs["noise_type"] = ntype
                        kwargs["strength"] = stren
                    elif op == "enhance":
                        kwargs["focus"] = foc
                    elif op == "scratch":
                        kwargs["severity"] = sev
                    return await _process_image(op, img, None, **kwargs)

                process_btn.click(
                    on_process,
                    inputs=[
                        input_image,
                        operation,
                        era_hint,
                        damage_type,
                        object_desc,
                        scale,
                        noise_type,
                        strength,
                        focus,
                        severity,
                    ],
                    outputs=[output_image, status_text],
                )

            # --- Remove Object Tab (with mask) ---
            with gr.TabItem("Remove Object (Masked)"):
                with gr.Row():
                    with gr.Column():
                        mask_image = gr.ImageEditor(
                            label="Upload Image & Draw Mask",
                            type="pil",
                            sources=["upload"],
                        )
                        mask_obj_desc = gr.Textbox(
                            label="Description of object to remove",
                            placeholder="e.g., the power line",
                        )
                        mask_process_btn = gr.Button("Remove Object", variant="primary")
                    with gr.Column():
                        mask_output = gr.Image(label="Result", type="pil")
                        mask_status = gr.Textbox(label="Status", interactive=False)

                async def on_mask_process(img_dict, desc):
                    if img_dict is None:
                        return Image.new("RGB", (1, 1)), "No image provided."
                    # Gradio ImageEditor returns dict with 'background' and 'layers'
                    if isinstance(img_dict, dict):
                        img = img_dict.get("background")
                        layers = img_dict.get("layers", [])
                        mask = layers[0] if layers else None
                    else:
                        img = img_dict
                        mask = None
                    return await _process_image(
                        "remove_object", img, mask, object_description=desc
                    )

                mask_process_btn.click(
                    on_mask_process,
                    inputs=[mask_image, mask_obj_desc],
                    outputs=[mask_output, mask_status],
                )

            # --- Batch Processing Tab ---
            with gr.TabItem("Batch Process"):
                batch_files = gr.File(
                    label="Upload Images",
                    file_count="multiple",
                    file_types=["image"],
                )
                batch_op = gr.Dropdown(
                    choices=[
                        ("Colorize", "colorize"),
                        ("Repair Damage", "repair"),
                        ("Upscale", "upscale"),
                        ("Denoise", "denoise"),
                        ("Enhance", "enhance"),
                        ("Remove Scratches", "scratch"),
                    ],
                    value="enhance",
                    label="Operation",
                )
                batch_btn = gr.Button("Process Batch", variant="primary")
                batch_gallery = gr.Gallery(label="Results")
                batch_status = gr.Textbox(label="Status", interactive=False)

                async def on_batch(files, op):
                    if not files:
                        return [], "No files uploaded."
                    results = []
                    messages = []
                    for f in files:
                        try:
                            img = Image.open(f.name)
                            out_img, msg = await _process_image(op, img, None)
                            results.append(out_img)
                            messages.append(f"{Path(f.name).name}: {msg}")
                        except Exception as exc:
                            messages.append(f"{Path(f.name).name}: Error - {exc}")
                    return results, "\n".join(messages)

                batch_btn.click(
                    on_batch,
                    inputs=[batch_files, batch_op],
                    outputs=[batch_gallery, batch_status],
                )

            # --- Gallery Tab ---
            with gr.TabItem("Gallery"):
                gallery_refresh = gr.Button("Refresh Gallery")
                gallery = gr.Gallery(label="Recent Outputs")
                gallery_status = gr.Textbox(label="Status", interactive=False)

                def refresh_gallery():
                    output_dir = Path("data/output")
                    if not output_dir.exists():
                        return [], "No outputs yet."
                    files = sorted(output_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
                    images = [str(f) for f in files[:50]]
                    return images, f"Showing {len(images)} recent outputs."

                gallery_refresh.click(refresh_gallery, outputs=[gallery, gallery_status])

            # --- Settings Tab ---
            with gr.TabItem("Settings"):
                gr.Markdown("### API Configuration")
                with gr.Row():
                    api_url = gr.Textbox(
                        label="NanoGPT API URL",
                        value=config.nanogpt.api_url,
                    )
                    api_key = gr.Textbox(
                        label="NanoGPT API Key",
                        value=config.nanogpt.api_key,
                        type="password",
                    )
                gr.Markdown("### Model Configuration")
                with gr.Row():
                    vision_model = gr.Textbox(
                        label="Vision Model (Prompt Enhancement)",
                        value=config.nanogpt.vision_model.model,
                    )
                    generation_model = gr.Textbox(
                        label="Generation Model (Image Output)",
                        value=config.nanogpt.generation_model.model,
                    )
                with gr.Row():
                    use_prompt_enhancement = gr.Checkbox(
                        label="Use GPT-4o Prompt Enhancement",
                        value=config.nanogpt.use_prompt_enhancement,
                    )
                gr.Markdown("### Processing Configuration")
                with gr.Row():
                    max_res = gr.Slider(
                        label="Max Resolution",
                        minimum=512,
                        maximum=4096,
                        step=64,
                        value=config.processing.max_resolution,
                    )
                    out_fmt = gr.Dropdown(
                        label="Output Format",
                        choices=["png", "jpg", "webp"],
                        value=config.processing.output_format,
                    )
                with gr.Row():
                    preserve_exif = gr.Checkbox(
                        label="Preserve EXIF Metadata",
                        value=config.processing.preserve_exif,
                    )
                    face_preserve = gr.Checkbox(
                        label="Face Preservation Checks",
                        value=config.processing.face_preservation,
                    )
                save_settings_btn = gr.Button("Save Settings", variant="primary")
                settings_status = gr.Textbox(label="Status", interactive=False)

                def save_settings(
                    url, key, vis_model, gen_model, prompt_enhance,
                    res, fmt, exif, face
                ):
                    cfg = get_config()
                    cfg.nanogpt.api_url = url
                    cfg.nanogpt.api_key = key
                    cfg.nanogpt.vision_model.model = vis_model
                    cfg.nanogpt.generation_model.model = gen_model
                    cfg.nanogpt.use_prompt_enhancement = prompt_enhance
                    cfg.processing.max_resolution = int(res)
                    cfg.processing.output_format = fmt
                    cfg.processing.preserve_exif = exif
                    cfg.processing.face_preservation = face
                    get_config_manager().save(cfg)
                    return "Settings saved successfully."

                save_settings_btn.click(
                    save_settings,
                    inputs=[
                        api_url, api_key, vision_model, generation_model,
                        use_prompt_enhancement, max_res, out_fmt,
                        preserve_exif, face_preserve,
                    ],
                    outputs=settings_status,
                )

    return app
