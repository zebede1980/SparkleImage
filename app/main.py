"""Main entry point for SparkleImage."""

import logging
import sys

from app.config.manager import get_config
from app.core.runtime import get_runtime
from app.ui.app import create_app


def setup_logging() -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> None:
    """Launch the SparkleImage Gradio application."""
    setup_logging()
    log = logging.getLogger("app.main")
    config = get_config()
    runtime = get_runtime()

    log.info("Local GPU: %s", config.comfy.url or "not configured")
    log.info("Face detector: %s; identity scoring: %s", runtime.face_backend,
             "on" if runtime.worker.scorer.available else "OFF — ArcFace model missing, faces won't be compared")
    if not config.comfy.url and not config.image.api_key:
        log.warning("Neither a local GPU nor a cloud API key is configured — set one in Settings.")
    if not config.auth_tuple:
        log.warning(
            "No UI authentication configured. Set SPARKLE_UI__AUTH_USERNAME and "
            "SPARKLE_UI__AUTH_PASSWORD before exposing this to the internet."
        )

    runtime.start()
    create_app().launch(
        server_name=config.ui.server_name,
        server_port=config.ui.server_port,
        share=config.ui.share,
        auth=config.auth_tuple,
        allowed_paths=[str(runtime.jobs.root.resolve())],
    )


if __name__ == "__main__":
    main()
