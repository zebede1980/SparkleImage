"""Gradio web UI for SparkleImage."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import gradio as gr
from PIL import Image

from app.config.manager import get_config
from app.core.gallery import recent
from app.core.operations import OPERATIONS, RESTORE_CHAIN, get_operation
from app.core.runtime import get_runtime, save_config_changes

logger = logging.getLogger(__name__)

INTRO = """# ✨ SparkleImage
Repair, colourise and clean up photographs — and get back *the same photograph*,
at its original size and shape, with the people in it still recognisable.
"""


def _status(result) -> str:
    """Render a RunResult's message and notes as markdown."""
    lines = [f"**{result.message}**" if result.success else f"⚠️ **{result.message}**"]
    lines += [f"- {note}" for note in result.notes]
    return "\n".join(lines)


def _mask_from_editor(value: Any) -> tuple[Optional[Image.Image], Optional[Image.Image]]:
    """Pull (photo, mask) out of a gr.ImageEditor value.

    The editor hands back the background plus the drawn layers; the mask is the
    alpha channel of what was painted.
    """
    if value is None:
        return None, None
    if not isinstance(value, dict):
        return value, None

    background = value.get("background")
    layers = value.get("layers") or []
    if background is None or not layers:
        return background, None

    mask = Image.new("L", background.size, 0)
    for layer in layers:
        if layer is None:
            continue
        layer_mask = layer.split()[-1] if layer.mode == "RGBA" else layer.convert("L")
        if layer_mask.size != mask.size:
            layer_mask = layer_mask.resize(mask.size, Image.NEAREST)
        mask = Image.composite(Image.new("L", mask.size, 255), mask, layer_mask)
    return background, (mask if mask.getbbox() else None)


def create_app() -> gr.Blocks:
    """Build and return the Gradio application."""
    runtime = get_runtime()
    config = get_config()

    with gr.Blocks(title="SparkleImage — AI photo restoration") as app:
        gr.Markdown(INTRO)

        with gr.Tabs():
            # ── Restore one photo ─────────────────────────────────────────────
            with gr.TabItem("Restore"):
                with gr.Row():
                    with gr.Column(scale=1):
                        single_input = gr.Image(
                            label="Photograph", type="pil", sources=["upload", "clipboard"]
                        )
                        single_op = gr.Dropdown(
                            choices=[
                                (op.label, op.id)
                                for op in OPERATIONS.values()
                                if not op.needs_mask
                            ],
                            value="enhance",
                            label="Operation",
                        )
                        single_help = gr.Markdown(get_operation("enhance").description)

                        # One control per parameter across all operations; only
                        # the selected operation's are shown.
                        param_inputs: dict[str, gr.components.Component] = {}
                        with gr.Group():
                            for op in OPERATIONS.values():
                                for param in op.params:
                                    key = f"{op.id}.{param.name}"
                                    if param.kind == "choice":
                                        component = gr.Dropdown(
                                            choices=list(param.choices),
                                            value=param.default,
                                            label=param.label,
                                            visible=False,
                                        )
                                    elif param.kind == "scale":
                                        component = gr.Slider(
                                            minimum=1, maximum=32, step=1,
                                            value=float(param.default),
                                            label=param.label,
                                            info=param.help,
                                            visible=False,
                                        )
                                    else:
                                        component = gr.Textbox(
                                            value=str(param.default),
                                            label=param.label,
                                            info=param.help,
                                            visible=False,
                                        )
                                    param_inputs[key] = component

                        single_run = gr.Button("Restore", variant="primary")

                    with gr.Column(scale=2):
                        single_output = gr.ImageSlider(
                            label="Before / after", type="pil", interactive=False
                        )
                        single_status = gr.Markdown()

                param_keys = list(param_inputs)
                param_components = [param_inputs[key] for key in param_keys]

                def _show_params(operation_id: str):
                    operation = get_operation(operation_id)
                    updates = [
                        gr.update(visible=key.startswith(f"{operation_id}.")) for key in param_keys
                    ]
                    return [operation.description, *updates]

                single_op.change(
                    _show_params, inputs=single_op, outputs=[single_help, *param_components]
                )

                async def _run_single(image, operation_id, *values):
                    if image is None:
                        return None, "Upload a photograph first."
                    params = {
                        key.split(".", 1)[1]: value
                        for key, value in zip(param_keys, values)
                        if key.startswith(f"{operation_id}.")
                    }
                    try:
                        result = await runtime.run(operation_id, image, params)
                    except Exception as exc:
                        logger.exception("Run failed")
                        return None, f"⚠️ **{exc}**"
                    return (image, result.image), _status(result)

                single_run.click(
                    _run_single,
                    inputs=[single_input, single_op, *param_components],
                    outputs=[single_output, single_status],
                )

            # ── Full restoration chain ────────────────────────────────────────
            with gr.TabItem("Full restore"):
                gr.Markdown(
                    "Runs several operations in order, each on the last one's output. "
                    "Damage is cleaned before colour is interpreted, and enlargement "
                    "comes last so the upscaler works on a repaired photograph."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        chain_input = gr.Image(label="Photograph", type="pil", sources=["upload", "clipboard"])
                        chain_steps = gr.CheckboxGroup(
                            choices=[(get_operation(step).label, step) for step in RESTORE_CHAIN],
                            value=["descratch", "repair", "colorize"],
                            label="Steps, in order",
                        )
                        chain_era = gr.Textbox(label="Era (optional)", placeholder="e.g. the 1950s")
                        chain_run = gr.Button("Restore fully", variant="primary")
                    with gr.Column(scale=2):
                        chain_output = gr.ImageSlider(label="Before / after", type="pil", interactive=False)
                        chain_status = gr.Markdown()

                async def _run_chain(image, steps, era, progress=gr.Progress()):
                    if image is None:
                        return None, "Upload a photograph first."
                    if not steps:
                        return None, "Pick at least one step."
                    ordered = [step for step in RESTORE_CHAIN if step in steps]

                    def on_step(index, total, operation):
                        progress((index - 1) / total, desc=f"{operation.label} ({index}/{total})")

                    try:
                        final, results = await runtime.restore_chain(
                            image, ordered, params={"colorize": {"era_hint": era}}, on_step=on_step
                        )
                    except Exception as exc:
                        logger.exception("Chain failed")
                        return None, f"⚠️ **{exc}**"

                    from app.core.gallery import save_result

                    path = save_result(
                        final, "restore",
                        output_format=config.processing.output_format,
                        jpeg_quality=config.processing.jpeg_quality,
                        metadata={"steps": ordered, "notes": [n for r in results for n in r.notes]},
                    )
                    lines = [f"**Restored in {len(ordered)} steps** — saved as {path.name}"]
                    for step, result in zip(ordered, results):
                        mark = "✅" if result.success else "⚠️"
                        lines.append(f"{mark} **{get_operation(step).label}** — {result.message}")
                        lines += [f"    - {note}" for note in result.notes]
                    return (image, final), "\n".join(lines)

                chain_run.click(
                    _run_chain,
                    inputs=[chain_input, chain_steps, chain_era],
                    outputs=[chain_output, chain_status],
                )

            # ── Mark and remove ───────────────────────────────────────────────
            with gr.TabItem("Mark & remove"):
                gr.Markdown(
                    "Paint over what you want gone. Only the painted area changes — "
                    "every other pixel of the photograph is left exactly as it was."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        mask_editor = gr.ImageEditor(
                            label="Paint over the object",
                            type="pil",
                            sources=["upload", "clipboard"],
                            brush=gr.Brush(colors=["#ff0000"], default_size=40),
                        )
                        mask_what = gr.Textbox(
                            label="What is it?", placeholder="e.g. the power line"
                        )
                        mask_run = gr.Button("Remove it", variant="primary")
                    with gr.Column(scale=2):
                        mask_output = gr.ImageSlider(label="Before / after", type="pil", interactive=False)
                        mask_status = gr.Markdown()

                async def _run_mask(value, description):
                    photo, mask = _mask_from_editor(value)
                    if photo is None:
                        return None, "Upload a photograph first."
                    if mask is None:
                        return None, "Paint over the area to remove first."
                    try:
                        result = await runtime.run(
                            "remove_object", photo,
                            {"object_description": description}, mask=mask,
                        )
                    except Exception as exc:
                        logger.exception("Removal failed")
                        return None, f"⚠️ **{exc}**"
                    return (photo, result.image), _status(result)

                mask_run.click(
                    _run_mask, inputs=[mask_editor, mask_what], outputs=[mask_output, mask_status]
                )

            # ── Batch ─────────────────────────────────────────────────────────
            with gr.TabItem("Batch"):
                batch_files = gr.File(
                    label="Photographs", file_count="multiple", file_types=["image"]
                )
                batch_op = gr.Dropdown(
                    choices=[
                        (op.label, op.id) for op in OPERATIONS.values() if not op.needs_mask
                    ],
                    value="enhance",
                    label="Operation",
                )
                batch_run = gr.Button("Run on all", variant="primary")
                batch_gallery = gr.Gallery(label="Results", columns=4, height=400)
                batch_status = gr.Markdown()

                async def _run_batch(files, operation_id, progress=gr.Progress()):
                    if not files:
                        return [], "Upload some photographs first."
                    outputs, lines = [], []
                    for index, item in enumerate(files, start=1):
                        name = Path(item.name).name
                        progress((index - 1) / len(files), desc=f"{name} ({index}/{len(files)})")
                        try:
                            with Image.open(item.name) as handle:
                                photo = handle.convert("RGB")
                            result = await runtime.run(operation_id, photo)
                            if result.success:
                                outputs.append(result.image)
                            lines.append(f"{'✅' if result.success else '⚠️'} {name} — {result.message}")
                        except Exception as exc:
                            lines.append(f"⚠️ {name} — {exc}")
                    return outputs, "\n\n".join(lines)

                batch_run.click(
                    _run_batch, inputs=[batch_files, batch_op], outputs=[batch_gallery, batch_status]
                )

            # ── Gallery ───────────────────────────────────────────────────────
            with gr.TabItem("Gallery"):
                gallery_refresh = gr.Button("Refresh")
                gallery_view = gr.Gallery(label="Saved results", columns=4, height=600)
                gallery_status = gr.Markdown()

                def _refresh_gallery():
                    files = recent()
                    return files, f"{len(files)} saved result(s)."

                gallery_refresh.click(_refresh_gallery, outputs=[gallery_view, gallery_status])
                app.load(_refresh_gallery, outputs=[gallery_view, gallery_status])

            # ── Settings ──────────────────────────────────────────────────────
            with gr.TabItem("Settings"):
                gr.Markdown("### Image API")
                with gr.Row():
                    set_base_url = gr.Textbox(label="API base URL", value=config.image.base_url)
                    set_api_key = gr.Textbox(
                        label="API key", value=config.image.api_key, type="password"
                    )
                fetch_models = gr.Button("Fetch models")
                fetch_status = gr.Markdown()

                gr.Markdown("### Models")
                set_edit_model = gr.Dropdown(
                    choices=[config.image.edit_model],
                    value=config.image.edit_model,
                    label="Edit model (image-to-image)",
                    info="Every restoration operation runs through this model",
                    allow_custom_value=True,
                )
                set_upscale_model = gr.Dropdown(
                    choices=[config.image.upscale_model],
                    value=config.image.upscale_model,
                    label="Upscale model",
                    info="A dedicated upscaler, not a generative model",
                    allow_custom_value=True,
                )
                set_creativity = gr.Slider(
                    minimum=-10, maximum=10, step=1, value=config.image.upscale_creativity,
                    label="Upscale creativity",
                    info="0 or below sharpens what is there; above 0 invents detail",
                )

                gr.Markdown("### Faces and output")
                with gr.Row():
                    set_face_crop = gr.Checkbox(
                        label="Re-edit faces at full resolution",
                        value=config.processing.face_crop_edit,
                        info=f"Detector in use: {runtime.face_backend}",
                    )
                    set_preserve_exif = gr.Checkbox(
                        label="Preserve EXIF", value=config.processing.preserve_exif
                    )
                with gr.Row():
                    set_format = gr.Dropdown(
                        choices=["png", "jpg", "webp"],
                        value=config.processing.output_format,
                        label="Output format",
                    )
                    set_max_upload = gr.Slider(
                        minimum=1, maximum=60, step=1,
                        value=config.processing.max_upload_megapixels,
                        label="Max upload megapixels",
                        info="Larger sources are downscaled for upload; output stays original size",
                    )

                save_button = gr.Button("Save settings", variant="primary")
                save_status = gr.Markdown()

                async def _fetch_models(base_url, api_key):
                    save_config_changes(**{"image.base_url": base_url, "image.api_key": api_key})
                    try:
                        models = await runtime.ensure_models(force=True)
                    except Exception as exc:
                        return (
                            f"⚠️ **Could not fetch models: {exc}**",
                            gr.update(), gr.update(),
                        )
                    edit_models = runtime.models_for("edit")
                    upscale_models = runtime.models_for("upscale")
                    summary = (
                        f"**{len(models)} image models.** "
                        f"{len(edit_models)} can edit, {len(upscale_models)} can upscale. "
                        "Content-filtered models are listed last — they reject photographs "
                        "of real people more often."
                    )
                    return (
                        summary,
                        gr.update(choices=[(runtime.model_label(m), m.id) for m in edit_models]),
                        gr.update(choices=[(runtime.model_label(m), m.id) for m in upscale_models]),
                    )

                fetch_models.click(
                    _fetch_models,
                    inputs=[set_base_url, set_api_key],
                    outputs=[fetch_status, set_edit_model, set_upscale_model],
                )

                def _save(base_url, api_key, edit_model, upscale_model, creativity,
                          face_crop, preserve_exif, output_format, max_upload):
                    save_config_changes(**{
                        "image.base_url": base_url,
                        "image.api_key": api_key,
                        "image.edit_model": edit_model,
                        "image.upscale_model": upscale_model,
                        "image.upscale_creativity": int(creativity),
                        "processing.face_crop_edit": face_crop,
                        "processing.preserve_exif": preserve_exif,
                        "processing.output_format": output_format,
                        "processing.max_upload_megapixels": float(max_upload),
                    })
                    return "Saved."

                save_button.click(
                    _save,
                    inputs=[set_base_url, set_api_key, set_edit_model, set_upscale_model,
                            set_creativity, set_face_crop, set_preserve_exif,
                            set_format, set_max_upload],
                    outputs=save_status,
                )

    return app
