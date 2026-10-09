"""Gallery tests."""

from pathlib import Path

from PIL import Image

from app.core.gallery import recent, save_result
from tests.conftest import make_photo


def test_filename_is_a_sortable_wall_clock_stamp(tmp_path: Path):
    path = save_result(make_photo((32, 32)), "colorize", output_dir=tmp_path)
    # e.g. 20260917-143000_colorize.png — not a monotonic clock reading.
    stamp = path.name.split("_")[0]
    assert len(stamp) == 15 and stamp[8] == "-"
    assert stamp[:4].startswith("20")


def test_two_saves_in_the_same_second_do_not_collide(tmp_path: Path):
    first = save_result(make_photo((32, 32)), "enhance", output_dir=tmp_path)
    second = save_result(make_photo((32, 32)), "enhance", output_dir=tmp_path)
    assert first != second and first.exists() and second.exists()


def test_jpeg_output_converts_and_applies_quality(tmp_path: Path):
    path = save_result(
        make_photo((64, 64)).convert("RGBA"), "repair", output_format="jpg", output_dir=tmp_path
    )
    assert path.suffix == ".jpg"
    assert Image.open(path).mode == "RGB"


def test_metadata_sidecar_records_what_was_run(tmp_path: Path):
    path = save_result(
        make_photo((32, 32)), "colorize",
        metadata={"model": "seedream-v4.5", "instruction": "Colourise this"},
        output_dir=tmp_path,
    )
    sidecar = path.with_suffix(path.suffix + ".json")
    assert sidecar.exists() and "seedream-v4.5" in sidecar.read_text()


def test_recent_lists_newest_first_and_ignores_sidecars(tmp_path: Path):
    save_result(make_photo((32, 32)), "one", output_dir=tmp_path, metadata={"a": 1})
    save_result(make_photo((32, 32)), "two", output_dir=tmp_path, metadata={"a": 2})
    listing = recent(output_dir=tmp_path)
    assert len(listing) == 2
    assert all(not name.endswith(".json") for name in listing)


def test_recent_on_a_missing_directory_is_empty(tmp_path: Path):
    assert recent(output_dir=tmp_path / "nope") == []
