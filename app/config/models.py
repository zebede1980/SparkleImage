"""Pydantic models for SparkleImage configuration."""

from typing import Optional
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class NanoGPTConfig(BaseModel):
    """NanoGPT API connection settings."""

    api_url: str = Field(
        default="https://api.nanogpt.com/v1",
        description="Base URL for the NanoGPT API",
    )
    api_key: str = Field(
        default="",
        description="API key for NanoGPT authentication",
    )
    model: str = Field(
        default="gpt-4o",
        description="Model identifier to use for image operations",
    )
    timeout: int = Field(
        default=120,
        ge=10,
        le=600,
        description="Request timeout in seconds",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum number of retries on failure",
    )

    @field_validator("api_url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("API URL must start with http:// or https://")
        return v.rstrip("/")


class ProcessingConfig(BaseModel):
    """Image processing parameters."""

    max_resolution: int = Field(
        default=2048,
        ge=512,
        le=4096,
        description="Maximum image dimension (width or height)",
    )
    output_format: str = Field(
        default="png",
        pattern=r"^(png|jpg|jpeg|webp)$",
        description="Output image format",
    )
    preserve_exif: bool = Field(
        default=True,
        description="Preserve EXIF metadata in output images",
    )
    face_preservation: bool = Field(
        default=True,
        description="Enable facial feature preservation checks",
    )
    default_quality: str = Field(
        default="high",
        pattern=r"^(low|medium|high|ultra)$",
        description="Default processing quality level",
    )
    jpeg_quality: int = Field(
        default=95,
        ge=1,
        le=100,
        description="JPEG compression quality (if output is JPEG)",
    )


class UIConfig(BaseModel):
    """Gradio web UI settings."""

    server_name: str = Field(
        default="0.0.0.0",
        description="Host to bind the Gradio server",
    )
    server_port: int = Field(
        default=7860,
        ge=1024,
        le=65535,
        description="Port for the Gradio server",
    )
    share: bool = Field(
        default=False,
        description="Create a public Gradio share link",
    )
    auth_username: Optional[str] = Field(
        default=None,
        description="Optional username for basic auth",
    )
    auth_password: Optional[str] = Field(
        default=None,
        description="Optional password for basic auth",
    )
    theme: str = Field(
        default="default",
        description="Gradio theme name",
    )


class AppConfig(BaseSettings):
    """Root application configuration."""

    model_config = SettingsConfigDict(
        env_prefix="SPARKLE_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    nanogpt: NanoGPTConfig = Field(default_factory=NanoGPTConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    ui: UIConfig = Field(default_factory=UIConfig)

    @property
    def auth_tuple(self) -> Optional[tuple[str, str]]:
        """Return auth tuple if both username and password are set."""
        if self.ui.auth_username and self.ui.auth_password:
            return (self.ui.auth_username, self.ui.auth_password)
        return None
