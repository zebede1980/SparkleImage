"""Main entry point for SparkleImage."""

import logging
import sys

from app.config.manager import get_config
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
    config = get_config()
    app = create_app()

    app.launch(
        server_name=config.ui.server_name,
        server_port=config.ui.server_port,
        share=config.ui.share,
        auth=config.auth_tuple,
    )


if __name__ == "__main__":
    main()
