import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from wig_recommender.api import app
from wig_recommender.face import MAX_BYTES, FaceError, FaceMeasurements, decode_image, measure_face
from wig_recommender.library import load_library
from wig_recommender.recommend import rank_wigs, recommend
from wig_recommender.shape import PROTOTYPES, SHAPES, shape_probabilities

PHOTO = Path.home() / "ai-avatar-platform/test-photo/bhargav.jpg"
needs_photo = pytest.mark.skipif(not PHOTO.exists(), reason="test photo missing")
client = TestClient(app)
WIGS = load_library()


def jpg(img) -> bytes:
    return cv2.imencode(".jpg", img)[1].tobytes()


def fake_face(vec) -> FaceMeasurements:
    return FaceMeasurements(*vec, cheek_px=200, box=(0, 0, 1, 1))


# ---------- image input ----------

@pytest.mark.parametrize("data,msg", [
    (b"", "empty"),
    (b"not an image at all", "not a readable image"),
    (b"\xff\xd8\xff\xe0" + b"\x00" * 500, "not a readable image"),  # truncated JPEG header
    (b"0" * (MAX_BYTES + 1), "larger than"),
])
def test_bad_bytes_rejected(data, msg):
    with pytest.raises(FaceError, match=msg):
        decode_image(data)


def test_blank_image_has_no_face():
    with pytest.raises(FaceError, match="no face"):
        measure_face(np.full((480, 640, 3), 255, np.uint8))


@needs_photo
def test_tiny_face_rejected():
    small = cv2.resize(cv2.imread(str(PHOTO)), (160, 90))
    with pytest.raises(FaceError, match="no face|too small"):
        measure_face(small)


@needs_photo
def test_two_faces_uses_largest_and_warns():
    img = cv2.imread(str(PHOTO))
    x, y, w, h = measure_face(img).box
    pad = w // 2  # crop around the face so both copies stay big enough to detect
    crop = img[max(0, y - pad): y + h + pad, max(0, x - pad): x + w + pad]
    small = cv2.resize(crop, None, fx=0.6, fy=0.6)
    canvas = np.zeros_like(crop)
    canvas[: small.shape[0], : small.shape[1]] = small
    m = measure_face(np.hstack([crop, canvas]))
    assert any("faces found" in w for w in m.warnings)
    assert m.box[0] < crop.shape[1]  # the big face is on the left half


# ---------- face shape ----------

@pytest.mark.parametrize("i,shape", list(enumerate(SHAPES)))
def test_each_prototype_classifies_as_itself(i, shape):
    probs = shape_probabilities(fake_face(PROTOTYPES[i]))
    assert max(probs, key=lambda s: probs[s]) == shape
    assert abs(sum(probs.values()) - 1) < 1e-3


def test_extreme_measurements_stay_valid():
    probs = shape_probabilities(fake_face([5.0, 0.1, 3.0, 0.0]))
    assert all(np.isfinite(v) and 0 <= v <= 1 for v in probs.values())


# ---------- ranking ----------

def test_long_face_prefers_fringe_and_not_long_hair():
    probs = {s: float(s == "oblong") for s in SHAPES}
    top = rank_wigs(probs, WIGS, top_k=3)
    assert all(r["wig"].length != "long" for r in top)
    assert top[0]["wig"].fringe == "full"


def test_round_face_avoids_chin_bob():
    probs = {s: float(s == "round") for s in SHAPES}
    ranked = rank_wigs(probs, WIGS, top_k=len(WIGS))
    assert ranked[-1]["wig"].length in ("bob", "pixie", "short")
    assert ranked[0]["wig"].length in ("medium", "long")


def test_filters_and_empty_result():
    probs = shape_probabilities(fake_face(PROTOTYPES[0]))
    assert all(r["wig"].texture == "curly" for r in rank_wigs(probs, WIGS, texture="curly"))
    assert rank_wigs(probs, WIGS, colour="neon green") == []
    with pytest.raises(ValueError):
        rank_wigs(probs, WIGS, top_k=0)


def test_manual_suitability_overrides_rules(tmp_path):
    data = json.loads(Path("data/wigs.json").read_text())
    data[0]["suitability"] = {s: 1.0 for s in SHAPES}
    p = tmp_path / "w.json"
    p.write_text(json.dumps(data))
    ranked = rank_wigs({s: 1 / 6 for s in SHAPES}, load_library(p), top_k=1)
    assert ranked[0]["wig"].id == data[0]["id"]


# ---------- library validation ----------

@pytest.mark.parametrize("content,msg", [
    ("[]", "non-empty"),
    ('{"id": "x"}', "non-empty"),
    ('[{"id": "a", "name": "A", "length": "huge", "fringe": "none", "texture": "wavy", "volume": "low"}]', "length"),
    ('[{"id": "a", "name": "A"}]', "wig #0"),
    ('[{"id": "a", "name": "A", "length": "bob", "fringe": "none", "texture": "wavy", "volume": "low", "colour": "x"}]', "wig #0"),
    ('[{"id": "a", "name": "A", "length": "bob", "fringe": "none", "texture": "wavy", "volume": "low", "suitability": {"oval": 2}}]', "0-1"),
])
def test_bad_library_rejected(tmp_path, content, msg):
    p = tmp_path / "w.json"
    p.write_text(content)
    with pytest.raises(ValueError, match=msg):
        load_library(p)


def test_duplicate_ids_rejected(tmp_path):
    data = json.loads(Path("data/wigs.json").read_text())
    p = tmp_path / "w.json"
    p.write_text(json.dumps(data + data[:1]))
    with pytest.raises(ValueError, match="duplicate"):
        load_library(p)


def test_corrupt_library_json(tmp_path):
    p = tmp_path / "w.json"
    p.write_text("{not json")
    with pytest.raises(json.JSONDecodeError):
        load_library(p)


# ---------- real photo, end to end ----------

@needs_photo
def test_real_photo_end_to_end():
    face, probs, ranked = recommend(PHOTO.read_bytes(), WIGS, top_k=3)
    assert 0.9 < face.length < 1.4
    assert len(ranked) == 3 and ranked[0]["score"] >= ranked[-1]["score"]


# ---------- HTTP API ----------

def test_api_health_and_wigs():
    assert client.get("/health").json()["wigs"] == len(WIGS)
    assert len(client.get("/wigs").json()) == len(WIGS)
    assert client.get("/wigs/quiff").status_code == 200


@pytest.mark.parametrize("wig_id", ["does-not-exist", "../etc/passwd", "x" * 500])
def test_api_unknown_wig_404(wig_id):
    assert client.get(f"/wigs/{wig_id}").status_code == 404


@pytest.mark.parametrize("data,code", [
    (b"", 400),
    (b"garbage", 400),
    (b"0" * (MAX_BYTES + 1), 413),
])
def test_api_bad_uploads(data, code):
    r = client.post("/recommend", files={"photo": ("x.jpg", data, "image/jpeg")})
    assert r.status_code == code


def test_api_no_face_422():
    r = client.post("/recommend", files={"photo": ("x.jpg", jpg(np.zeros((300, 300, 3), np.uint8)), "image/jpeg")})
    assert r.status_code == 422


def test_api_missing_file_and_bad_params():
    assert client.post("/recommend").status_code == 422
    files = {"photo": ("x.jpg", b"x", "image/jpeg")}
    assert client.post("/recommend?top_k=0", files=files).status_code == 422
    assert client.post("/recommend?length=huge", files=files).status_code == 422


@needs_photo
def test_api_real_photo():
    r = client.post("/recommend?top_k=2", files={"photo": ("p.jpg", PHOTO.read_bytes(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["face"]["shape"] in SHAPES
    assert len(body["recommendations"]) == 2
    assert "suitability" not in body["recommendations"][0]["wig"]
