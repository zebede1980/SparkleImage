"""What the Jobs tab shows: summaries, choices, previews, face strips."""

import io

import pytest
from PIL import Image

from app.core.jobs import JobOptions, JobStore
from app.ui import views
from tests.conftest import make_photo


@pytest.fixture
def job_with_candidates(tmp_path):
    store = JobStore(tmp_path)
    buf = io.BytesIO()
    make_photo((640, 480)).save(buf, "JPEG")
    job = store.create(buf.getvalue(), "wedding.jpg", JobOptions())
    for cid in ("local-s1", "local-s2"):
        make_photo((320, 240), seed=len(cid)).save(store.dir(job.id) / f"cand_{cid}.png")
    job.status, job.stage, job.mode, job.size = "done", "Done", "colourise", (640, 480)
    job.faces = [{"index": 0, "box": [40, 40, 60, 60], "landmarks": None, "frame": [320, 240]},
                 {"index": 1, "box": [200, 60, 60, 60], "landmarks": None, "frame": [320, 240]}]
    job.candidates = [
        {"id": "local-s1", "engine": "x", "seed": 1, "is_local": True, "error": "", "scores": [0.85, 0.31], "mean": 0.58},
        {"id": "local-s2", "engine": "x", "seed": 2, "is_local": True, "error": "", "scores": [0.80, 0.42], "mean": 0.61},
    ]
    job.base = "local-s2"
    job.face_choices = {"0": "local-s1"}
    store.save(job)
    return store, job


def test_summary_names_the_base_and_flags_faces_too_faint_to_judge(job_with_candidates):
    _, job = job_with_candidates
    text = views.job_summary(job)
    assert "Base: **local-s2**" in text
    assert "1 face(s) taken from other candidates" in text
    assert "Face(s) 2 have too little detail" in text  # best 0.42 < 0.5; face 1 (0.85) is fine


def test_face_options_offer_the_base_first_then_scored_alternatives(job_with_candidates):
    _, job = job_with_candidates
    options = views.face_options(job, 0)
    assert options[0] == ("Base's face (local-s2, 0.80)", "")
    assert ("local-s1 (0.85)", "local-s1") in options


def test_face_options_follow_an_unapplied_base_change(job_with_candidates):
    _, job = job_with_candidates
    options = views.face_options(job, 0, base_id="local-s1")
    assert options[0] == ("Base's face (local-s1, 0.85)", "")
    assert ("local-s2 (0.80)", "local-s2") in options
    assert all(v != "local-s1" for _, v in options[1:])


def test_previews_are_cached_and_phone_sized(job_with_candidates):
    store, job = job_with_candidates
    big = store.dir(job.id) / "restored.png"
    make_photo((4000, 3000)).save(big)
    path = views.preview(big)
    assert max(Image.open(path).size) == views.PREVIEW_EDGE
    assert views.preview(big) == path
    assert views.preview(None) is None


def test_face_strip_has_the_original_plus_one_tile_per_candidate(job_with_candidates):
    store, job = job_with_candidates
    strip = Image.open(views.face_strip(store, job, 0))
    assert strip.width == 3 * (views.FACE_TILE + 6)
    assert views.face_strip(store, job, 5) is None


def test_gallery_marks_the_base(job_with_candidates):
    store, job = job_with_candidates
    captions = [caption for _, caption in views.candidate_gallery(store, job)]
    assert any("BASE" in c and c.startswith("local-s2") for c in captions)
