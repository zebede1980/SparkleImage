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

    log.info("Edit model: %s", config.image.edit_model)
    log.info("Upscale model: %s", config.image.upscale_model)
    log.info("Face detector: %s", runtime.face_backend)
    if not config.image.api_key:
        log.warning("No API key configured — set one in the Settings tab before restoring anything.")
    if not config.auth_tuple:
        log.warning(
            "No UI authentication configured. Set SPARKLE_UI__AUTH_USERNAME and "
            "SPARKLE_UI__AUTH_PASSWORD before exposing this to the internet."
        )

    create_app().launch(
        server_name=config.ui.server_name,
        server_port=config.ui.server_port,
        share=config.ui.share,
        auth=config.auth_tuple,
    )


if __name__ == "__main__":
    main()
