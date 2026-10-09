"""The restoration pipeline, with fake engines and a fake identity scorer."""

import pytest
from PIL import Image, ImageOps

from app.core import restore
from app.core.identity import SourceFace
from app.core.faces import FaceBox
from app.core.restore import Candidate
from tests.conftest import make_photo


class FakeEngine:
    def __init__(self, name="local/fake", is_local=True, fail_seeds=()):
        self.name, self.is_local, self.fail_seeds = name, is_local, set(fail_seeds)
        self.calls = []

    async def edit(self, image, instruction, seed):
        self.calls.append(seed)
        if seed in self.fail_seeds:
            raise RuntimeError("GPU on fire")
        return make_photo((320, 240), seed=seed)

    async def upscale(self, image, short_edge):
        self.upscaled_to = short_edge
        scale = short_edge / min(image.size)
        return image.resize((round(image.width * scale), round(image.height * scale)))


def cand(cid, scores, local=True, size=(320, 240)):
    return Candidate(id=cid, engine="x", seed=0, is_local=local, image=make_photo(size), scores=scores)


FACES = [SourceFace(0, FaceBox(40, 40, 60, 60), None, (320, 240)), SourceFace(1, FaceBox(200, 60, 60, 60), None, (320, 240))]


class TestModeAndInstruction:
    def test_black_and_white_is_colourised(self):
        assert restore.choose_mode(ImageOps.grayscale(make_photo()).convert("RGB")) == "colourise"

    def test_colour_is_corrected(self):
        assert restore.choose_mode(make_photo()) == "correct"

    def test_every_instruction_protects_the_people(self):
        for mode in ("colourise", "correct", "keep_bw"):
            assert "Keep every person exactly the same" in restore.instruction_for(mode)

    def test_era_hint_goes_into_the_colourise_instruction(self):
        assert "the 1940s" in restore.instruction_for("colourise", "the 1940s")


class TestGenerate:
    @pytest.mark.asyncio
    async def test_one_candidate_per_seed_plus_cloud(self):
        local, cloud = FakeEngine(), FakeEngine("seedream-v4.5", is_local=False)
        out = await restore.generate(make_photo(), "do it", local, [1, 2, 3], [cloud])
        assert [c.id for c in out] == ["local-s1", "local-s2", "local-s3", "cloud-seedream-v4.5"]
        assert local.calls == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_a_failed_candidate_is_recorded_not_raised(self):
        out = await restore.generate(make_photo(), "do it", FakeEngine(fail_seeds=[2]), [1, 2])
        assert out[0].ok and not out[1].ok
        assert "GPU on fire" in out[1].error

    @pytest.mark.asyncio
    async def test_cloud_only(self):
        out = await restore.generate(make_photo(), "x", None, [7], [FakeEngine("m", is_local=False)])
        assert [c.id for c in out] == ["cloud-m"]


class TestChoosing:
    def test_base_is_the_best_local_even_if_a_cloud_model_scores_higher(self):
        cands = [cand("local-s1", [0.80, 0.85]), cand("local-s2", [0.84, 0.86]), cand("cloud-x", [0.95, 0.95], local=False)]
        assert restore.default_base(cands) == "local-s2"

    def test_failed_candidates_are_never_the_base(self):
        broken = Candidate(id="local-s1", engine="x", seed=1, is_local=True, error="boom")
        assert restore.default_base([broken, cand("local-s2", [0.5])]) == "local-s2"

    def test_faces_are_swapped_in_only_past_the_margin(self):
        cands = [cand("local-s1", [0.80, 0.90]), cand("local-s2", [0.90, 0.91])]
        choices = restore.default_faces(cands, "local-s1", FACES, margin=0.02)
        assert choices == {0: "local-s2"}  # +0.10 swaps; +0.01 doesn't

    def test_a_face_missing_from_the_base_is_filled_from_whoever_has_it(self):
        cands = [cand("local-s1", [None, 0.9]), cand("local-s2", [0.7, 0.8])]
        assert restore.default_faces(cands, "local-s1", FACES) == {0: "local-s2"}


class TestCompose:
    def test_no_choices_returns_the_base(self):
        cands = [cand("local-s1", [0.8, 0.8])]
        out = restore.compose(cands, "local-s1", {}, FACES)
        assert out.tobytes() == cands[0].image.tobytes()

    def test_a_base_at_another_size_gets_faces_in_the_right_place(self):
        big = cand("cloud-x", [0.9, 0.9], local=False, size=(640, 480))
        donor = Candidate(id="local-s1", engine="x", seed=0, is_local=True,
                          image=Image.new("RGB", (320, 240), (0, 0, 255)), scores=[0.95, 0.9])
        out = restore.compose([big, donor], "cloud-x", {0: "local-s1"}, FACES)
        assert out.size == (640, 480)
        # Face 0 is at (40..100) in a 320-wide frame, so (80..200) here: changed there, not at the far side.
        assert out.getpixel((140, 140)) != big.image.getpixel((140, 140))
        assert out.getpixel((600, 450)) == big.image.getpixel((600, 450))


class TestFinish:
    @pytest.mark.asyncio
    async def test_restored_is_upscaled_but_never_shrunk(self):
        engine = FakeEngine()
        out = await restore.finish_restored(engine, make_photo((320, 240)), 2160)
        assert min(out.size) == 2160
        await restore.finish_restored(engine, make_photo((4000, 3000)), 2160)
        assert engine.upscaled_to == 3000

    def test_faithful_is_at_the_scans_resolution(self):
        scan = ImageOps.grayscale(make_photo((1000, 750))).convert("RGB")
        assert restore.finish_faithful(scan, make_photo((320, 240))).size == (1000, 750)
