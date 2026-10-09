"""Gradio web UI for SparkleImage.

Restore submits photos as jobs and returns at once; Jobs shows them live.
Nothing a restoration produces lives in the browser — close the tab, lock the
phone, come back tomorrow: it's all in data/jobs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import gradio as gr
from PIL import Image

from app.config.manager import get_config
from app.core.jobs import JobOptions
from app.core.runtime import get_runtime, save_config_changes
from app.ui import views

logger = logging.getLogger(__name__)

INTRO = """# ✨ SparkleImage
Old photographs repaired, colourised and sharpened — with the people in them still themselves.
"""

# Faces shown for per-face choices. Group photos rarely have more people whose
# faces are large enough to matter; the rest still come from the base image.
MAX_FACES = 8

# Offered on the Restore tab. In testing these two were the cloud models worth
# their few pence; others drifted further from the people in the photo.
CLOUD_SUGGESTIONS = ["seedream-v4.5", "flux-2-max-image-to-image"]

MODES = [
    ("Automatic", "auto"),
    ("Colourise", "colourise"),
    ("Correct colour", "correct"),
    ("Keep black & white", "keep_bw"),
]


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
    store = runtime.jobs
    config = get_config()

    def job_choices() -> list[tuple[str, str]]:
        return [(views.job_label(job), job.id) for job in store.all()[:40]]

    def fingerprint(job_id: str) -> str:
        """Changes whenever anything the job view shows would change."""
        try:
            job = store.load(job_id)
        except FileNotFoundError:
            return ""
        return f"{job.status}|{job.stage}|{len(job.log)}|{job.base}|{sorted(job.face_choices.items())}"

    with gr.Blocks(title="SparkleImage — photo restoration") as app:
        gr.Markdown(INTRO)
        gpu_status = gr.Markdown()

        with gr.Tabs() as tabs:
            # ── Restore ───────────────────────────────────────────────────────
            with gr.TabItem("Restore", id="restore"):
                files = gr.File(label="Photographs", file_count="multiple", file_types=["image"])
                mode = gr.Radio(MODES, value="auto", label="What to do",
                                info="Automatic colourises black & white and corrects faded colour")
                era = gr.Textbox(label="Era (optional)", placeholder="e.g. the 1940s — helps colourising")
                seeds = gr.Slider(1, 8, step=1, value=config.restore.local_seeds, label="Candidates",
                                  info="Each is a separate attempt (~25s on the local GPU). More attempts, "
                                       "better odds every face comes out right")
                cloud = gr.CheckboxGroup(
                    choices=list(dict.fromkeys(CLOUD_SUGGESTIONS + config.restore.cloud_models)),
                    value=config.restore.cloud_models,
                    label="Also try cloud models (nano-gpt, a few pence each)",
                )
                submit = gr.Button("Restore", variant="primary")
                submit_status = gr.Markdown()

            # ── Jobs ──────────────────────────────────────────────────────────
            with gr.TabItem("Jobs", id="jobs"):
                timer = gr.Timer(4)
                seen = gr.State("")
                with gr.Row():
                    with gr.Column(scale=1, min_width=260):
                        job_list = gr.Radio(choices=job_choices(), label="Jobs", type="value")
                    with gr.Column(scale=3):
                        summary = gr.Markdown("Pick a job.")
                        which = gr.Radio(
                            [("Restored", "restored"), ("Faithful — the scan's own detail, AI colour", "faithful")],
                            value="restored", label="Show",
                        )
                        slider = gr.ImageSlider(label="Before / after", type="filepath", interactive=False)
                        downloads = gr.File(label="Full size", file_count="multiple", interactive=False)
                        with gr.Accordion("Candidates and faces", open=False):
                            gr.Markdown(
                                "Every candidate is a separate restoration. Each face is scored for likeness to "
                                "the original (1.00 = identical); the best version of each face is used automatically. "
                                "Change the picks here if you disagree, then press **Use these choices** — only the "
                                "upscale is re-run."
                            )
                            gallery = gr.Gallery(label="Candidates", columns=4, height="auto")
                            base_choice = gr.Dropdown(label="Base image", choices=[])
                            face_groups, face_images, face_choices = [], [], []
                            for i in range(MAX_FACES):
                                with gr.Group(visible=False) as group:
                                    face_images.append(gr.Image(show_label=False, interactive=False, type="filepath"))
                                    face_choices.append(gr.Dropdown(label=f"Face {i + 1}", choices=[]))
                                face_groups.append(group)
                            apply = gr.Button("Use these choices", variant="primary")
                        with gr.Accordion("Log", open=False):
                            log = gr.Textbox(lines=10, show_label=False, interactive=False)
                        delete = gr.Button("Delete this job", variant="stop", size="sm")

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
                        mask_what = gr.Textbox(label="What is it?", placeholder="e.g. the power line")
                        mask_run = gr.Button("Remove it", variant="primary")
                    with gr.Column(scale=2):
                        mask_output = gr.ImageSlider(label="Before / after", type="pil", interactive=False)
                        mask_status = gr.Markdown()

            # ── Settings ──────────────────────────────────────────────────────
            with gr.TabItem("Settings"):
                gr.Markdown("### Local GPU (ComfyUI via comfy-gateway)")
                with gr.Row():
                    set_comfy_url = gr.Textbox(label="Gateway URL", value=config.comfy.url)
                    set_comfy_key = gr.Textbox(label="Gateway API key", value=config.comfy.api_key, type="password")
                test_gpu = gr.Button("Test connection")
                with gr.Row():
                    set_edit_mp = gr.Slider(0.5, 2.0, step=0.1, value=config.comfy.edit_megapixels,
                                            label="Edit size (megapixels)",
                                            info="1.5 takes ~25s on 16GB; above 2 spills VRAM and takes minutes")
                    set_short_edge = gr.Slider(1024, 3072, step=64, value=config.comfy.upscale_short_edge,
                                               label="Upscale to (short side, pixels)")

                gr.Markdown("### Restoring")
                with gr.Row():
                    set_seeds = gr.Slider(1, 8, step=1, value=config.restore.local_seeds, label="Default candidates")
                    set_margin = gr.Slider(0.0, 0.1, step=0.005, value=config.restore.face_swap_margin,
                                           label="Face swap margin",
                                           info="A face is taken from another candidate only if it's this much more alike")
                set_cloud = gr.Textbox(label="Default cloud models (comma-separated)",
                                       value=", ".join(config.restore.cloud_models))

                gr.Markdown("### Cloud (nano-gpt)")
                with gr.Row():
                    set_base_url = gr.Textbox(label="API base URL", value=config.image.base_url)
                    set_api_key = gr.Textbox(label="API key", value=config.image.api_key, type="password")
                set_edit_model = gr.Textbox(label="Model for Mark & remove", value=config.image.edit_model)
                set_preserve_exif = gr.Checkbox(label="Preserve EXIF (Mark & remove)", value=config.processing.preserve_exif)
                save_button = gr.Button("Save settings", variant="primary")
                save_status = gr.Markdown()

        # ── Wiring ───────────────────────────────────────────────────────────

        def _submit(paths, mode_value, era_value, seed_count, cloud_models):
            if not paths:
                return "Add some photographs first.", gr.skip(), gr.skip(), gr.skip()
            options = JobOptions(mode=mode_value, era=era_value or "", seeds=int(seed_count), cloud_models=list(cloud_models or []))
            submitted = []
            for path in paths:
                job = runtime.worker.submit(Path(path).read_bytes(), Path(path).name, options)
                submitted.append(job)
            message = f"Queued {len(submitted)} photo(s). They run one at a time — you can close this page."
            return message, gr.Tabs(selected="jobs"), gr.Radio(choices=job_choices(), value=submitted[-1].id), None

        submit.click(_submit, inputs=[files, mode, era, seeds, cloud],
                     outputs=[submit_status, tabs, job_list, files])

        def _render(job_id, which_value):
            empty = [gr.skip()] * (8 + 3 * MAX_FACES)
            if not job_id:
                return ["Pick a job.", ""] + empty
            try:
                job = store.load(job_id)
            except FileNotFoundError:
                return ["That job has gone.", ""] + empty

            after_name = "faithful.jpg" if which_value == "faithful" else "restored.png"
            after = views.preview(store.image(job, after_name)) or views.preview(store.image(job, "composite.png"))
            before = views.source_preview(store, job)
            full = [str(p) for p in (store.image(job, "restored.png"), store.image(job, "faithful.jpg")) if p]

            base_options = [(f"{c['id']}" + (f" (likeness {c['mean']:.2f})" if c.get("mean") is not None else ""), c["id"])
                            for c in job.candidates if store.image(job, f"cand_{c['id']}.png")]
            groups, images, dropdowns = [], [], []
            for i in range(MAX_FACES):
                face = next((f for f in job.faces if f["index"] == i), None)
                groups.append(gr.Group(visible=face is not None))
                images.append(views.face_strip(store, job, i) if face else None)
                dropdowns.append(gr.Dropdown(choices=views.face_options(job, i), value=job.face_choices.get(str(i), ""))
                                 if face else gr.Dropdown(choices=[], value=None))
            return [
                views.job_summary(job),
                fingerprint(job_id),
                (before, after) if before and after else None,
                full or None,
                views.candidate_gallery(store, job),
                gr.Dropdown(choices=base_options, value=job.base or None),
                "\n".join(job.log),
                gr.Button(interactive=job.status == "done"),
                gr.Radio(),
                gr.skip(),
                *groups, *images, *dropdowns,
            ]

        job_outputs = [summary, seen, slider, downloads, gallery, base_choice, log, apply, which, gpu_status,
                       *face_groups, *face_images, *face_choices]
        job_list.change(_render, inputs=[job_list, which], outputs=job_outputs)
        which.change(_render, inputs=[job_list, which], outputs=job_outputs)

        def _tick(job_id, which_value, last_seen):
            # Refresh the list always; re-render the open job only if it changed,
            # so a half-made choice in a dropdown isn't wiped every four seconds.
            listing = gr.Radio(choices=job_choices(), value=job_id or None)
            if job_id and fingerprint(job_id) != last_seen:
                return [listing] + _render(job_id, which_value)
            return [listing] + [gr.skip()] * len(job_outputs)

        timer.tick(_tick, inputs=[job_list, which, seen], outputs=[job_list, *job_outputs])

        def _apply(job_id, base_value, *picks):
            if not job_id:
                return gr.skip()
            choices = {str(i): pick for i, pick in enumerate(picks) if pick}
            runtime.worker.rechoose(job_id, base=base_value, face_choices=choices)
            return "Re-composited; upscaling again…"

        apply.click(_apply, inputs=[job_list, base_choice, *face_choices], outputs=summary)

        def _delete(job_id):
            if job_id:
                job = store.load(job_id)
                if job.status == "running":
                    return gr.skip(), "That job is running — delete it once it has finished."
                store.delete(job_id)
            return gr.Radio(choices=job_choices(), value=None), "Deleted."

        delete.click(_delete, inputs=job_list, outputs=[job_list, summary])

        async def _gpu_status():
            try:
                return await runtime.local_gpu_status()
            except Exception as exc:
                return f"Local GPU: {exc}"

        app.load(_gpu_status, outputs=gpu_status)
        test_gpu.click(_gpu_status, outputs=gpu_status)

        async def _run_mask(value, description):
            photo, mask = _mask_from_editor(value)
            if photo is None:
                return None, "Upload a photograph first."
            if mask is None:
                return None, "Paint over the area to remove first."
            try:
                result = await runtime.run("remove_object", photo, {"object_description": description}, mask=mask)
            except Exception as exc:
                logger.exception("Removal failed")
                return None, f"⚠️ **{exc}**"
            return (photo, result.image), _status(result)

        mask_run.click(_run_mask, inputs=[mask_editor, mask_what], outputs=[mask_output, mask_status])

        def _save(comfy_url, comfy_key, edit_mp, short_edge, seed_count, margin, cloud_text,
                  base_url, api_key, edit_model, preserve_exif):
            try:
                save_config_changes(**{
                    "comfy.url": comfy_url.strip(),
                    "comfy.api_key": comfy_key.strip(),
                    "comfy.edit_megapixels": float(edit_mp),
                    "comfy.upscale_short_edge": int(short_edge),
                    "restore.local_seeds": int(seed_count),
                    "restore.face_swap_margin": float(margin),
                    "restore.cloud_models": [m.strip() for m in cloud_text.split(",") if m.strip()],
                    "image.base_url": base_url.strip(),
                    "image.api_key": api_key.strip(),
                    "image.edit_model": edit_model.strip(),
                    "processing.preserve_exif": preserve_exif,
                })
            except Exception as exc:
                return f"⚠️ {exc}"
            return "Saved. New jobs use these settings."

        save_button.click(
            _save,
            inputs=[set_comfy_url, set_comfy_key, set_edit_mp, set_short_edge, set_seeds, set_margin, set_cloud,
                    set_base_url, set_api_key, set_edit_model, set_preserve_exif],
            outputs=save_status,
        )

    return app
