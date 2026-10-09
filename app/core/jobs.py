"""Restoration jobs: persisted on disk, run one at a time by a background worker.

A restoration takes minutes — several candidates at ~25s each plus an
upscale — so it can't live inside a browser request. A phone locking, a tab
closing or a dropped connection must not lose anything (CardGenV2 learned
this the hard way). So the browser only ever submits work and reads state;
the worker owns the work, and everything it produces is on disk:

    data/jobs/<id>/
        job.json            state, scores, choices, progress log
        source.<ext>        the upload, byte for byte
        cand_<id>.png       each candidate restoration
        composite.png       base + chosen faces
        restored.png        composite, SeedVR2-upscaled
        faithful.jpg        the scan's own detail, coloured from the composite

There is one GPU, so one worker. A restart re-queues unfinished jobs, and a
re-run skips candidates already on disk, so no GPU time is spent twice.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from PIL import Image

from app.core import restore
from app.core.engines import Engine, LocalEngine
from app.core.faces import FaceBox
from app.core.identity import IdentityScorer, SourceFace
from app.core.restore import Candidate

logger = logging.getLogger(__name__)

Status = Literal["queued", "running", "done", "failed"]
Action = Literal["restore", "finish"]


@dataclass
class JobOptions:
    mode: str = "auto"  # auto | colourise | correct | keep_bw
    era: str = ""
    seeds: int = 4
    cloud_models: list[str] = field(default_factory=list)


@dataclass
class Job:
    id: str
    photo: str
    created: float
    options: JobOptions
    status: Status = "queued"
    action: Action = "restore"
    stage: str = "Waiting for the GPU"
    log: list[str] = field(default_factory=list)
    error: str = ""
    mode: str = ""
    instruction: str = ""
    size: tuple[int, int] = (0, 0)
    faces: list[dict] = field(default_factory=list)       # {index, box: [x,y,w,h], landmarks, frame}
    candidates: list[dict] = field(default_factory=list)  # {id, engine, seed, is_local, error, scores, mean}
    base: str = ""
    face_choices: dict[str, str] = field(default_factory=dict)
    finished: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Job":
        data = json.loads(text)
        data["options"] = JobOptions(**data.get("options", {}))
        data["size"] = tuple(data.get("size", (0, 0)))
        return cls(**data)

    def source_faces(self) -> list[SourceFace]:
        """Faces as the compositor needs them — boxes only; embeddings stay in the scores."""
        return [
            SourceFace(f["index"], FaceBox(*f["box"], landmarks=tuple(map(tuple, f["landmarks"])) if f.get("landmarks") else None),
                       None, tuple(f["frame"]))
            for f in self.faces
        ]


class JobStore:
    """Jobs on disk. Every write is whole-file and atomic, so a crash never leaves half a job.json."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def dir(self, job_id: str) -> Path:
        return self.root / job_id

    def create(self, photo_bytes: bytes, filename: str, options: JobOptions) -> Job:
        job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        suffix = Path(filename).suffix.lower() or ".jpg"
        folder = self.dir(job_id)
        folder.mkdir(parents=True)
        (folder / f"source{suffix}").write_bytes(photo_bytes)
        job = Job(id=job_id, photo=Path(filename).name, created=time.time(), options=options)
        self.save(job)
        return job

    def save(self, job: Job) -> None:
        with self._lock:
            path = self.dir(job.id) / "job.json"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(job.to_json(), encoding="utf-8")
            tmp.replace(path)

    def load(self, job_id: str) -> Job:
        return Job.from_json((self.dir(job_id) / "job.json").read_text(encoding="utf-8"))

    def all(self) -> list[Job]:
        jobs = []
        for path in self.root.glob("*/job.json"):
            try:
                jobs.append(Job.from_json(path.read_text(encoding="utf-8")))
            except Exception as exc:  # one corrupt job mustn't hide the rest
                logger.warning("Skipping unreadable job %s: %s", path.parent.name, exc)
        return sorted(jobs, key=lambda j: j.created, reverse=True)

    def source(self, job: Job) -> Image.Image:
        path = next(self.dir(job.id).glob("source.*"))
        return restore.load_photo(Image.open(path))

    def image(self, job: Job, name: str) -> Optional[Path]:
        path = self.dir(job.id) / name
        return path if path.exists() else None

    def candidates(self, job: Job) -> list[Candidate]:
        """Candidates with their images loaded from disk."""
        out = []
        for c in job.candidates:
            path = self.dir(job.id) / f"cand_{c['id']}.png"
            out.append(Candidate(id=c["id"], engine=c["engine"], seed=c["seed"], is_local=c["is_local"],
                                 image=Image.open(path).convert("RGB") if path.exists() else None,
                                 error=c.get("error", ""), scores=c.get("scores", [])))
        return out

    def delete(self, job_id: str) -> None:
        import shutil

        shutil.rmtree(self.dir(job_id), ignore_errors=True)


@dataclass
class Engines:
    """What a job runs on, built fresh from config for each job so Settings changes apply."""

    local: Optional[LocalEngine]
    cloud: list[Engine]
    short_edge: int
    face_margin: float


class Worker:
    """Runs jobs one at a time on its own thread and event loop.

    Gradio's request handlers come and go; the worker doesn't. Submitting is
    thread-safe and returns immediately.
    """

    def __init__(self, store: JobStore, engines: Callable[[JobOptions], Engines],
                 scorer: Optional[IdentityScorer] = None) -> None:
        self.store = store
        self.engines = engines
        self.scorer = scorer or IdentityScorer()
        self._queue: Optional[asyncio.Queue[str]] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ready = threading.Event()

    # ── lifecycle ───────────────────────────────────────────────────────────

    def start(self) -> None:
        threading.Thread(target=self._run_loop, name="sparkle-worker", daemon=True).start()
        self._ready.wait(10)
        for job in reversed(self.store.all()):  # oldest first
            if job.status in ("queued", "running"):
                logger.info("Re-queuing job %s after restart", job.id)
                job.status, job.stage = "queued", "Waiting for the GPU (re-queued after a restart)"
                self.store.save(job)
                self.enqueue(job.id)

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._queue = asyncio.Queue()
        self._ready.set()
        self._loop.run_until_complete(self._consume())

    async def _consume(self) -> None:
        assert self._queue is not None
        while True:
            job_id = await self._queue.get()
            try:
                await self.process(job_id)
            except Exception:  # pragma: no cover - process() records its own failures
                logger.exception("Worker crashed on %s", job_id)

    def enqueue(self, job_id: str) -> None:
        assert self._loop is not None and self._queue is not None, "worker not started"
        self._loop.call_soon_threadsafe(self._queue.put_nowait, job_id)

    # ── public operations (called from the UI thread) ───────────────────────

    def submit(self, photo_bytes: bytes, filename: str, options: JobOptions) -> Job:
        job = self.store.create(photo_bytes, filename, options)
        self.enqueue(job.id)
        return job

    def rechoose(self, job_id: str, base: Optional[str] = None, face_choices: Optional[dict[str, str]] = None) -> Job:
        """Change the base and/or per-face choices, re-composite now (CPU), queue the upscale."""
        job = self.store.load(job_id)
        if base:
            job.base = base
        if face_choices is not None:
            job.face_choices = {k: v for k, v in face_choices.items() if v and v != job.base}
        self._compose(job)
        job.status, job.action, job.stage = "queued", "finish", "Waiting for the GPU to upscale"
        self.store.save(job)
        self.enqueue(job.id)
        return job

    # ── the work ────────────────────────────────────────────────────────────

    def _note(self, job: Job, message: str, stage: Optional[str] = None) -> None:
        job.log.append(f"{time.strftime('%H:%M:%S')} {message}")
        if stage:
            job.stage = stage
        self.store.save(job)

    def _compose(self, job: Job) -> Image.Image:
        candidates = self.store.candidates(job)
        choices = {int(k): v for k, v in job.face_choices.items()}
        composite = restore.compose(candidates, job.base, choices, job.source_faces())
        composite.save(self.store.dir(job.id) / "composite.png")
        return composite

    async def process(self, job_id: str) -> None:
        job = self.store.load(job_id)
        job.status = "running"
        self.store.save(job)
        try:
            engines = self.engines(job.options)
            if job.action == "restore":
                await self._restore(job, engines)
            await self._finish(job, engines)
            job.status, job.stage, job.finished = "done", "Done", time.time()
            self._note(job, "finished")
        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            job.status, job.error, job.stage = "failed", str(exc), "Failed"
            self._note(job, f"failed: {exc}")

    async def _restore(self, job: Job, engines: Engines) -> None:
        source = self.store.source(job)
        job.size = source.size
        mode = job.options.mode if job.options.mode != "auto" else restore.choose_mode(source)
        job.mode, job.instruction = mode, restore.instruction_for(mode, job.options.era)
        self._note(job, f"{source.width}x{source.height}, mode {mode}", "Restoring")

        existing = {c.id: c for c in self.store.candidates(job) if c.ok}
        seeds = list(range(1, job.options.seeds + 1))
        todo_seeds = [s for s in seeds if f"local-s{s}" not in existing]
        todo_cloud = [e for e in engines.cloud if f"cloud-{e.name.replace('/', '_')}" not in existing]
        total = len(seeds) + len(engines.cloud)
        finished = [len(existing)]

        def progress(message: str) -> None:
            finished[0] += 1
            self._note(job, message, f"Restoring — {finished[0]}/{total} candidates")

        fresh = await restore.generate(source, job.instruction, engines.local if todo_seeds else None,
                                       todo_seeds, todo_cloud, progress)
        for c in fresh:
            if c.ok:
                c.image.save(self.store.dir(job.id) / f"cand_{c.id}.png")
        candidates = list(existing.values()) + fresh
        if not any(c.ok for c in candidates):
            raise RuntimeError("Every candidate failed: " + "; ".join(c.error for c in candidates if c.error))

        self._note(job, "scoring faces", "Comparing faces")
        faces = restore.score(self.scorer, source, candidates)
        job.faces = [{"index": f.index, "box": [f.box.x, f.box.y, f.box.width, f.box.height],
                      "landmarks": [list(p) for p in f.box.landmarks] if f.box.landmarks else None,
                      "frame": list(f.frame)} for f in faces]
        job.candidates = [{"id": c.id, "engine": c.engine, "seed": c.seed, "is_local": c.is_local,
                           "error": c.error, "scores": c.scores, "mean": c.mean} for c in candidates]
        job.base = restore.default_base(candidates) or ""
        job.face_choices = {str(k): v for k, v in restore.default_faces(candidates, job.base, faces, engines.face_margin).items()}
        self._note(job, f"{len(faces)} faces; base {job.base}; swaps {job.face_choices or 'none'}")
        self._compose(job)

    async def _finish(self, job: Job, engines: Engines) -> None:
        composite = Image.open(self.store.dir(job.id) / "composite.png").convert("RGB")
        source = self.store.source(job)
        restore.finish_faithful(source, composite).save(self.store.dir(job.id) / "faithful.jpg", quality=95)
        if engines.local is None:
            self._note(job, "no local GPU configured — skipped the upscale")
            composite.save(self.store.dir(job.id) / "restored.png")
            return
        self._note(job, "upscaling", "Upscaling")
        restored = await restore.finish_restored(engines.local, composite, engines.short_edge)
        restored.save(self.store.dir(job.id) / "restored.png")
