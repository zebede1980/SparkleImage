"""Jobs: persistence, the worker, resuming, and re-choosing."""

import asyncio
import io

import pytest
from PIL import Image

from app.core.jobs import Engines, Job, JobOptions, JobStore, Worker
from tests.conftest import make_photo
from tests.test_identity import FakeEmbedder, FixedDetector, face_at
from app.core.identity import IdentityScorer


class FakeLocal:
    name, is_local = "local/fake", True

    def __init__(self, fail=False):
        self.edits, self.upscales, self.fail = [], 0, fail

    async def edit(self, image, instruction, seed):
        if self.fail:
            raise RuntimeError("PC is off")
        self.edits.append(seed)
        return make_photo((320, 240), seed=seed * 40)

    async def upscale(self, image, short_edge):
        self.upscales += 1
        return image.resize((image.width * 2, image.height * 2))


def photo_bytes() -> bytes:
    buf = io.BytesIO()
    make_photo((640, 480)).save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture
def setup(tmp_path):
    store = JobStore(tmp_path / "jobs")
    local = FakeLocal()
    scorer = IdentityScorer(FixedDetector([face_at(40, 40, 60), face_at(200, 60, 60)], (320, 240)), FakeEmbedder())
    worker = Worker(store, lambda options: Engines(local, [], 480, 0.02), scorer)
    return store, worker, local


def run(worker, job_id):
    asyncio.run(worker.process(job_id))


class TestStore:
    def test_round_trip(self, tmp_path):
        store = JobStore(tmp_path)
        job = store.create(photo_bytes(), "Grandad 1942.jpg", JobOptions(seeds=2, era="the 1940s"))
        loaded = store.load(job.id)
        assert loaded.photo == "Grandad 1942.jpg"
        assert loaded.options.era == "the 1940s"
        assert store.source(loaded).size == (640, 480)

    def test_newest_first_and_corrupt_jobs_skipped(self, tmp_path):
        store = JobStore(tmp_path)
        a = store.create(photo_bytes(), "a.jpg", JobOptions())
        b = store.create(photo_bytes(), "b.jpg", JobOptions())
        b.created = a.created + 10
        store.save(b)
        (tmp_path / "broken").mkdir()
        (tmp_path / "broken" / "job.json").write_text("{not json")
        assert [j.photo for j in store.all()] == ["b.jpg", "a.jpg"]


class TestWorker:
    def test_a_job_runs_to_done_with_every_output(self, setup):
        store, worker, local = setup
        job = store.create(photo_bytes(), "p.jpg", JobOptions(seeds=3))
        run(worker, job.id)
        job = store.load(job.id)
        assert job.status == "done", job.error
        assert local.edits == [1, 2, 3] and local.upscales == 1
        assert len(job.faces) == 2 and job.base.startswith("local-s")
        for name in ("composite.png", "restored.png", "faithful.jpg", "cand_local-s1.png"):
            assert store.image(job, name), name
        assert Image.open(store.image(job, "faithful.jpg")).size == (640, 480)

    def test_a_rerun_reuses_candidates_already_on_disk(self, setup):
        store, worker, local = setup
        job = store.create(photo_bytes(), "p.jpg", JobOptions(seeds=2))
        run(worker, job.id)
        job = store.load(job.id)
        job.status, job.action = "queued", "restore"
        store.save(job)
        local.edits.clear()
        run(worker, job.id)
        assert local.edits == []  # nothing regenerated
        assert store.load(job.id).status == "done"

    def test_when_every_candidate_fails_the_job_fails_with_the_reason(self, setup, tmp_path):
        store = JobStore(tmp_path / "j2")
        worker = Worker(store, lambda o: Engines(FakeLocal(fail=True), [], 480, 0.02), setup[1].scorer)
        job = store.create(photo_bytes(), "p.jpg", JobOptions(seeds=2))
        run(worker, job.id)
        job = store.load(job.id)
        assert job.status == "failed" and "PC is off" in job.error

    def test_rechoosing_recomposites_and_only_reupscales(self, setup):
        store, worker, local = setup
        job = store.create(photo_bytes(), "p.jpg", JobOptions(seeds=2))
        run(worker, job.id)
        before = store.image(store.load(job.id), "composite.png").read_bytes()
        worker.enqueue = lambda job_id: None  # run the finish by hand below
        job = worker.rechoose(job.id, base="local-s2", face_choices={"0": "local-s1"})
        assert job.base == "local-s2" and job.face_choices == {"0": "local-s1"}
        assert store.image(job, "composite.png").read_bytes() != before
        local.edits.clear()
        run(worker, job.id)
        assert local.edits == [] and local.upscales == 2
        assert store.load(job.id).status == "done"

    def test_choosing_the_base_as_a_face_donor_is_dropped(self, setup):
        store, worker, _ = setup
        job = store.create(photo_bytes(), "p.jpg", JobOptions(seeds=2))
        run(worker, job.id)
        worker.enqueue = lambda job_id: None
        job = worker.rechoose(job.id, base="local-s1", face_choices={"0": "local-s1", "1": ""})
        assert job.face_choices == {}

    def test_start_requeues_unfinished_jobs(self, setup):
        store, worker, _ = setup
        stuck = store.create(photo_bytes(), "p.jpg", JobOptions(seeds=1))
        stuck.status = "running"
        store.save(stuck)
        queued = []
        worker._run_loop = lambda: (setattr(worker, "_loop", object()), setattr(worker, "_queue", object()), worker._ready.set())
        worker.enqueue = queued.append
        worker.start()
        assert queued == [stuck.id]
        assert store.load(stuck.id).status == "queued"
