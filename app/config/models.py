"""Pydantic models for SparkleImage configuration."""

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class ImageAPIConfig(BaseModel):
    """Image API connection and model selection."""

    base_url: str = Field(
        default="https://nano-gpt.com/api/v1",
        description="OpenAI-compatible API root used for image calls",
    )
    api_key: str = Field(default="", description="API key")
    timeout: int = Field(
        default=300,
        ge=30,
        le=900,
        description="Request timeout in seconds — a high-resolution edit can run past a minute",
    )
    max_retries: int = Field(default=2, ge=0, le=5)

    edit_model: str = Field(
        default="seedream-v4.5",
        description=(
            "Image-to-image model used for every restoration operation. "
            "seedream-v4.5 is the default because it is not content-filtered, "
            "accepts free-form output resolutions and goes up to 16.8MP"
        ),
    )
    upscale_model: str = Field(
        default="clarity-ai-pro-upscaler",
        description="Dedicated upscaler — never a generative edit model",
    )
    upscale_creativity: int = Field(
        default=0,
        ge=-10,
        le=10,
        description=(
            "How much detail the upscaler may invent. Zero or below sharpens "
            "what is there, which is the only honest setting for a photograph "
            "of a real person"
        ),
    )
    warn_on_filtered_models: bool = Field(
        default=True,
        description="Warn when the selected model applies a content filter",
    )

    @field_validator("base_url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("API URL must start with http:// or https://")
        return v.rstrip("/")


class ProcessingConfig(BaseModel):
    """Image processing parameters."""

    output_format: str = Field(default="png", pattern=r"^(png|jpg|jpeg|webp)$")
    jpeg_quality: int = Field(default=95, ge=1, le=100)
    preserve_exif: bool = Field(
        default=True, description="Carry EXIF metadata from the source into the output"
    )
    composite_feather: int = Field(
        default=8,
        ge=0,
        le=64,
        description="Blur radius on composite edges, in pixels, to hide the seam",
    )
    max_upload_megapixels: float = Field(
        default=24.0,
        ge=1.0,
        le=100.0,
        description="Downscale sources larger than this before uploading",
    )


class ComfyConfig(BaseModel):
    """The home PC's ComfyUI, reached through comfy-gateway."""

    url: str = Field(default="", description="Gateway URL, e.g. https://comfy.example.com; empty disables local")
    api_key: str = Field(default="", description="The gateway's X-API-Key")
    edit_megapixels: float = Field(
        default=1.5,
        ge=0.5,
        le=2.0,
        description="Size Qwen-Image edits run at. ~25s at 1.5MP on 16GB; above 2MP it spills VRAM and takes minutes",
    )
    upscale_short_edge: int = Field(
        default=2160, ge=512, le=4096, description="SeedVR2 output size, in pixels on the short side"
    )
    run_timeout: int = Field(default=600, ge=60, le=3600)

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("ComfyUI URL must start with http:// or https://")
        return v.rstrip("/")


class RestoreConfig(BaseModel):
    """How a restoration job builds its candidates."""

    local_seeds: int = Field(
        default=4,
        ge=1,
        le=12,
        description="Local candidates per photo. Seeds vary identity as much as models do; ~25s each",
    )
    cloud_models: list[str] = Field(
        default_factory=list,
        description="nano-gpt models to add as extra candidates, e.g. seedream-v4.5 (a few pence each)",
    )
    face_swap_margin: float = Field(
        default=0.02,
        ge=0.0,
        le=0.2,
        description="A face is swapped in from another candidate only if it scores this much better",
    )


class UIConfig(BaseModel):
    """Gradio web UI settings."""

    server_name: str = Field(default="0.0.0.0")
    server_port: int = Field(default=7860, ge=1024, le=65535)
    share: bool = Field(default=False)
    auth_username: Optional[str] = Field(default=None)
    auth_password: Optional[str] = Field(default=None)
    theme: str = Field(default="default")


class AppConfig(BaseSettings):
    """Root application configuration."""

    model_config = SettingsConfigDict(
        env_prefix="SPARKLE_",
        env_nested_delimiter="__",
        # docker-compose passes `${VAR:-}` as an empty string when VAR is unset;
        # without this, that empty string would override what Settings saved.
        env_ignore_empty=True,
        extra="ignore",
        yaml_file="data/config.yaml",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Environment variables outrank the saved YAML file.

        The old loader read the YAML with `model_validate`, which bypasses the
        settings machinery entirely — so every SPARKLE_* override in
        docker-compose worked on first boot and was silently ignored from the
        second onwards, once the app had written data/config.yaml.
        """
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )

    image: ImageAPIConfig = Field(default_factory=ImageAPIConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    comfy: ComfyConfig = Field(default_factory=ComfyConfig)
    restore: RestoreConfig = Field(default_factory=RestoreConfig)
    ui: UIConfig = Field(default_factory=UIConfig)

    @property
    def auth_tuple(self) -> Optional[tuple[str, str]]:
        """Return auth tuple if both username and password are set."""
        if self.ui.auth_username and self.ui.auth_password:
            return (self.ui.auth_username, self.ui.auth_password)
        return None
