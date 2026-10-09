"""Identity scoring: matching faces across candidates, and the ArcFace embedding itself."""

import numpy as np
import pytest
from PIL import Image

from app.core.faces import FaceBox, FaceDetector
from app.core.identity import FaceEmbedder, IdentityScorer, _match, mean_score
from tests.conftest import make_photo

MARKS = ((40.0, 50.0), (70.0, 50.0), (55.0, 65.0), (43.0, 85.0), (67.0, 85.0))


def face_at(x: int, y: int, size: int = 100) -> FaceBox:
    scale = size / 112
    marks = tuple((x + px * scale, y + py * scale) for px, py in MARKS)
    return FaceBox(x, y, size, size, 0.9, marks)


class FixedDetector(FaceDetector):
    """Reports the same faces in whatever image it's shown, scaled to that image."""

    def __init__(self, faces, frame):
        super().__init__()
        self.faces, self.frame = faces, frame

    @property
    def gives_landmarks(self):
        return True

    def detect(self, image):
        fx, fy = image.width / self.frame[0], image.height / self.frame[1]
        return [f.scaled(fx, fy) for f in self.faces]


class FakeEmbedder(FaceEmbedder):
    """Embeds a face as the mean colour under its landmarks — enough to tell faces apart."""

    def __init__(self):
        super().__init__(model_path=None)

    @property
    def available(self):
        return True

    def embed(self, rgb, landmarks):
        x, y = (int(v) for v in landmarks[2])
        v = rgb[max(0, y - 5):y + 5, max(0, x - 5):x + 5].reshape(-1, 3).mean(axis=0).astype(np.float64) + 1
        return v / np.linalg.norm(v)


class TestMatch:
    def test_nearest_face_within_reach_is_matched(self):
        source = face_at(100, 100)
        assert _match(source, [face_at(400, 100), face_at(110, 105)]) == face_at(110, 105)

    def test_a_face_too_far_away_is_not_the_same_person(self):
        assert _match(face_at(100, 100), [face_at(400, 400)]) is None

    def test_nothing_to_match(self):
        assert _match(face_at(100, 100), []) is None


class TestScorer:
    def make(self, frame=(400, 300)):
        faces = [face_at(50, 50), face_at(250, 100)]
        return IdentityScorer(FixedDetector(faces, frame), FakeEmbedder())

    def test_source_faces_are_ordered_left_to_right_and_carry_their_frame(self):
        scorer = self.make()
        faces = scorer.source_faces(make_photo((800, 600)), (400, 300))
        assert [f.index for f in faces] == [0, 1]
        assert faces[0].box.x < faces[1].box.x
        assert all(f.frame == (400, 300) for f in faces)

    def test_an_unchanged_candidate_scores_perfectly(self):
        scorer = self.make()
        source = make_photo((400, 300))
        faces = scorer.source_faces(source, (400, 300))
        assert scorer.score(source, faces) == pytest.approx([1.0, 1.0])

    def test_a_candidate_at_another_size_is_scored_in_the_source_frame(self):
        scorer = self.make()
        source = make_photo((400, 300))
        faces = scorer.source_faces(source, (400, 300))
        scores = scorer.score(source.resize((800, 600)), faces)
        assert scores == pytest.approx([1.0, 1.0], abs=0.01)

    def test_a_missing_face_is_none_not_zero(self):
        frame = (400, 300)
        scorer = self.make(frame)
        faces = scorer.source_faces(make_photo(frame), frame)
        scorer.detector = FixedDetector([face_at(50, 50)], frame)  # second person gone
        scores = scorer.score(make_photo(frame), faces)
        assert scores[0] is not None and scores[1] is None

    def test_unavailable_without_landmarks(self):
        class NoMarks(FaceDetector):
            gives_landmarks = False

        assert not IdentityScorer(NoMarks(), FakeEmbedder()).available


def test_mean_ignores_missing_faces():
    assert mean_score([0.8, None, 0.6]) == pytest.approx(0.7)
    assert mean_score([None]) is None
    assert mean_score([]) is None


@pytest.mark.skipif(not FaceEmbedder().available, reason="ArcFace model not in data/models")
class TestRealArcFace:
    def test_embedding_is_unit_length_and_512d(self):
        rgb = np.asarray(make_photo((300, 300)))
        v = FaceEmbedder().embed(rgb, face_at(100, 100).landmarks)
        assert v.shape == (512,)
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-4)

    def test_the_same_face_is_more_similar_than_a_different_one(self):
        embedder = FaceEmbedder()
        a = np.asarray(make_photo((300, 300), seed=1))
        b = np.asarray(make_photo((300, 300), seed=1).transpose(Image.Transpose.ROTATE_180))
        marks = face_at(60, 60).landmarks
        same = embedder.embed(a, marks) @ embedder.embed(a.copy(), marks)
        different = embedder.embed(a, marks) @ embedder.embed(b, marks)
        assert same == pytest.approx(1.0, abs=1e-4)
        assert different < same
