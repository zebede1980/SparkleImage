"""ComfyUI client against a mock transport, and the workflow graphs."""

import io
import json

import httpx
import pytest
from PIL import Image

from app.clients import workflows
from app.clients.comfy import ComfyClient, ComfyError, ComfyOffline
from tests.conftest import make_photo


def png_bytes(size=(64, 48)) -> bytes:
    buf = io.BytesIO()
    make_photo(size).save(buf, "PNG")
    return buf.getvalue()


class FakeComfy:
    """Just enough ComfyUI: upload, prompt, history (pending a few times), view."""

    def __init__(self, pending_polls=2, fail_with=None, stall_polls=0):
        self.pending, self.fail_with, self.stalls = pending_polls, fail_with, stall_polls
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/upload/image":
            return httpx.Response(200, json={"name": "a.png", "subfolder": "sparkle"})
        if path == "/prompt":
            return httpx.Response(200, json={"prompt_id": "p1"})
        if path.startswith("/history/"):
            if self.stalls:
                self.stalls -= 1
                raise httpx.ConnectError("stalled mid model swap")
            if self.pending:
                self.pending -= 1
                return httpx.Response(200, json={})
            if self.fail_with:
                status = {"status_str": "error", "completed": False,
                          "messages": [["execution_error", {"exception_message": self.fail_with}]]}
                return httpx.Response(200, json={"p1": {"status": status, "outputs": {}}})
            out = {"56": {"images": [{"filename": "r.png", "subfolder": "sparkle", "type": "output"}]}}
            return httpx.Response(200, json={"p1": {"status": {"status_str": "success", "completed": True}, "outputs": out}})
        if path == "/view":
            return httpx.Response(200, content=png_bytes())
        return httpx.Response(404)


def client(fake, **kw) -> ComfyClient:
    return ComfyClient("https://gpu.example", "secret", poll_interval=0, transport=httpx.MockTransport(fake), **kw)


class TestRun:
    @pytest.mark.asyncio
    async def test_upload_run_and_fetch(self):
        fake = FakeComfy()
        async with client(fake) as c:
            name = await c.upload(make_photo())
            image = await c.run({"56": {}}, "56")
        assert name == "sparkle/a.png"
        assert image.size == (64, 48)
        assert all(r.headers["X-API-Key"] == "secret" for r in fake.requests)

    @pytest.mark.asyncio
    async def test_stalled_polls_are_survived(self):
        # ComfyUI's server stalls during model swaps; that isn't a failed job.
        async with client(FakeComfy(stall_polls=3)) as c:
            assert (await c.run({}, "56")).size == (64, 48)

    @pytest.mark.asyncio
    async def test_an_execution_error_is_reported_with_comfys_message(self):
        async with client(FakeComfy(fail_with="CUDA out of memory")) as c:
            with pytest.raises(ComfyError, match="CUDA out of memory"):
                await c.run({}, "56")

    @pytest.mark.asyncio
    async def test_a_job_that_never_finishes_times_out(self):
        async with client(FakeComfy(pending_polls=10**6), run_timeout=0.05) as c:
            with pytest.raises(ComfyError, match="too long"):
                await c.run({}, "56")

    @pytest.mark.asyncio
    async def test_a_bad_key_says_so(self):
        transport = httpx.MockTransport(lambda r: httpx.Response(403))
        async with ComfyClient("https://gpu.example", "wrong", transport=transport) as c:
            with pytest.raises(ComfyError, match="API key"):
                await c.queue({})

    @pytest.mark.asyncio
    async def test_unreachable_is_offline(self):
        def down(request):
            raise httpx.ConnectError("no route to host")

        async with ComfyClient("https://gpu.example", "k", transport=httpx.MockTransport(down)) as c:
            with pytest.raises(ComfyOffline):
                await c.queue({})
            assert not await c.available()

    @pytest.mark.asyncio
    async def test_a_rejected_workflow_passes_comfys_reason_on(self):
        transport = httpx.MockTransport(lambda r: httpx.Response(400, text='{"error": "model not found: x.safetensors"}'))
        async with ComfyClient("https://gpu.example", "k", transport=transport) as c:
            with pytest.raises(ComfyError, match="x.safetensors"):
                await c.queue({})


class TestWorkflows:
    def test_qwen_edit_wires_source_prompt_and_seed(self):
        graph = workflows.qwen_edit("sparkle/a.png", "fix it", seed=11, megapixels=1.5)
        assert graph["78"]["inputs"]["image"] == "sparkle/a.png"
        assert graph["64"]["inputs"]["prompt"] == "fix it"
        assert graph["3"]["inputs"]["seed"] == 11
        assert graph["79"]["inputs"]["resize_type.megapixels"] == 1.5
        assert graph[workflows.OUTPUT_NODE]["class_type"] == "SaveImage"
        json.dumps(graph)  # must be serialisable as-is

    def test_every_link_points_at_a_node_that_exists(self):
        for graph in (workflows.qwen_edit("a", "b", 1), workflows.seedvr2_upscale("a", 2160)):
            for node in graph.values():
                for value in node["inputs"].values():
                    if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                        assert value[0] in graph

    def test_seedvr2_resolution_is_even(self):
        assert workflows.seedvr2_upscale("a", 2161)["4"]["inputs"]["resolution"] == 2160
