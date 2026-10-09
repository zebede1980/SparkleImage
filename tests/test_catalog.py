"""Catalogue tests, against a trimmed copy of a real GET /api/models response."""

import json
from pathlib import Path

import pytest

from app.clients.catalog import ImageModel, catalogue_url, parse_image_models

FIXTURE = Path(__file__).parent / "fixtures" / "catalogue.json"


@pytest.fixture(scope="module")
def models() -> dict[str, ImageModel]:
    return parse_image_models(json.loads(FIXTURE.read_text()))


class TestCatalogueUrl:
    def test_strips_the_openai_compatible_suffix(self):
        # /v1/models lists text models only — the image catalogue is one level up.
        assert catalogue_url("https://nano-gpt.com/api/v1") == "https://nano-gpt.com/api/models"
        assert catalogue_url("https://nano-gpt.com/api/v1/") == "https://nano-gpt.com/api/models"

    def test_leaves_a_plain_root_alone(self):
        assert catalogue_url("https://example.com/api") == "https://example.com/api/models"


class TestCapabilityDerivation:
    def test_both_means_generate_and_edit(self, models):
        seedream = models["seedream-v4.5"]
        assert seedream.icon_label == "both"
        assert seedream.can_generate and seedream.can_edit

    def test_image_to_image_cannot_generate(self, models):
        edit_only = models["flux-2-pro-image-to-image"]
        assert edit_only.can_edit and not edit_only.can_generate

    def test_text_to_image_cannot_edit(self, models):
        gen_only = models["flux-2-pro"]
        assert gen_only.can_generate and not gen_only.can_edit

    def test_combine_comes_from_the_declared_flag_not_the_name(self, models):
        # CardGenV2 has to default this off for everything because nothing in a
        # model id signals multi-reference support; the catalogue states it.
        assert models["qwen-image-3-pro"].can_combine
        assert not models["reve/2.1/edit"].can_combine

    def test_missing_icon_label_falls_back_to_the_name(self, models):
        # 15 older entries carry no iconLabel at all; every one of them is a
        # plain text-to-image model, so the name fallback must not promote them
        # to edit-capable. dall-e-3 is the fitting example: the model this app
        # was originally built on genuinely cannot edit an image.
        dalle = models["dall-e-3"]
        assert dalle.icon_label is None
        assert dalle.can_generate and not dalle.can_edit

    def test_name_fallback_still_catches_edit_shaped_ids(self, models):
        unlabelled = ImageModel(id="some-vendor/mystery-kontext", name="Mystery")
        assert unlabelled.can_edit and not unlabelled.can_generate

    def test_upscalers_are_identified_by_their_parameters(self, models):
        assert models["clarity-ai-pro-upscaler"].can_upscale
        assert models["seedvr2-image"].can_upscale
        assert models["Upscaler"].can_upscale
        assert not models["seedream-v4.5"].can_upscale


class TestModelMetadata:
    def test_seedream_45_is_unfiltered(self, models):
        # The single most important attribute for restoring photos of real
        # people: a filtered model returns a blank frame instead of an error.
        assert models["seedream-v4.5"].is_filtered is False
        assert models["seedream-v5.0-lite"].is_filtered is True
        assert models["reve/2.1/edit"].is_filtered is True

    def test_custom_resolution_is_parsed(self, models):
        limits = models["seedream-v4.5"].custom_resolution
        assert limits is not None
        assert (limits.min_width, limits.max_width, limits.step) == (1024, 4096, 64)
        assert limits.min_total_pixels == 3_686_400

    def test_models_without_custom_resolution_report_none(self, models):
        assert models["flux-2-pro-image-to-image"].custom_resolution is None

    def test_resolution_presets_split_from_opaque_tokens(self, models):
        seedream = models["seedream-v4.5"]
        assert (4096, 4096) in seedream.presets
        assert "auto" not in seedream.presets
        # "auto" is a provider label, not a size — it must survive as a token.
        assert "auto" in models["clarity-ai-pro-upscaler"].resolution_tokens

    def test_input_size_limit_is_read(self, models):
        assert models["seedream-v4.5"].max_input_bytes == 10 * 1024 * 1024

    def test_max_megapixels(self, models):
        assert models["seedream-v4.5"].max_megapixels == pytest.approx(16.78, abs=0.1)


class TestParsing:
    def test_an_empty_or_wrong_shaped_payload_yields_nothing(self):
        assert parse_image_models({}) == {}
        assert parse_image_models({"models": {"text": {"gpt": {}}}}) == {}

    def test_non_dict_entries_are_skipped(self):
        parsed = parse_image_models({"models": {"image": {"a": "nonsense", "b": {"name": "B"}}}})
        assert list(parsed) == ["b"]
