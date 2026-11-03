from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import wig_recommender.api as api
from wig_recommender.face import FaceError, find_landmarks
from wig_recommender.fit import (
    EYES, _include_thin_strands, fill_from_background, fit_wig, has_image, placement,
)
from wig_recommender.library import load_library
from wig_recommender.tryon import FACE_OVAL

PHOTO = Path.home() / "ai-avatar-platform/test-photo/bhargav.jpg"
needs_photo = pytest.mark.skipif(not PHOTO.exists(), reason="test photo missing")
needs_wig = pytest.mark.skipif(not has_image("side-part"), reason="side-part image not generated")
WIGS = load_library()
client = TestClient(api.app)


def post(url, data):
    return client.post(url, files={"photo": ("p.jpg", data, "image/jpeg")})


# ---------- placement ----------

def test_placement_scales_and_shifts():
    pts = np.zeros((468, 2), np.float32)
    anchors = {"cheek_l": [0, 0], "cheek_r": [100, 0], "forehead": [50, -60], "chin": [50, 120]}
    for k, i in {"cheek_l": 234, "cheek_r": 454, "forehead": 10, "chin": 152}.items():
        pts[i] = np.array(anchors[k]) * 2 + [300, 400]  # same face, twice as big, moved
    m = placement(anchors, pts)
    assert np.allclose(m @ [100, 0, 1], [500, 400], atol=1)
    assert np.allclose(np.hypot(*m[:, 0]), 2, atol=0.01)  # scale 2, no rotation


# ---------- strand mask and background fill ----------

def test_strand_mask_adds_thin_dark_lines_but_not_wide_objects():
    img = np.full((240, 300, 3), 180, np.uint8)
    cv2.line(img, (138, 108), (138, 135), (20, 20, 20), 2)
    cv2.rectangle(img, (145, 98), (173, 132), (20, 20, 20), -1)
    erase = np.zeros(img.shape[:2], np.uint8)
    erase[100:140, 90:130] = 255
    empty = np.zeros_like(erase)
    result = _include_thin_strands(img, erase, empty, empty, empty.astype(bool), 100)
    assert result[118, 138] == 255
    assert not result[98:133, 145:174].any()


def test_strand_mask_threshold_12_catches_fainter_line():
    img = np.full((180, 240, 3), 125, np.uint8)
    cv2.line(img, (118, 75), (118, 105), (110, 110, 110), 1)
    erase = np.zeros(img.shape[:2], np.uint8)
    erase[75:105, 90:110] = 255
    empty = np.zeros_like(erase)
    at_20 = _include_thin_strands(img, erase, empty, empty, empty.astype(bool), 100, 20)
    at_12 = _include_thin_strands(img, erase, empty, empty, empty.astype(bool), 100, 12)
    assert at_20[90, 118] == 0
    assert at_12[90, 118] == 255


def test_fit_defaults_to_faint_strand_threshold():
    import inspect

    assert inspect.signature(fit_wig).parameters["blackhat_threshold"].default == 12


def test_strand_mask_excludes_face_head_and_wig():
    img = np.full((180, 240, 3), 180, np.uint8)
    cv2.line(img, (118, 75), (118, 105), (20, 20, 20), 2)
    erase = np.zeros(img.shape[:2], np.uint8)
    erase[75:105, 90:110] = 255
    face = np.zeros_like(erase)
    face[65:115, 112:145] = 255
    head = np.zeros_like(erase)
    head[60:120, 110:160] = 255
    wig = np.zeros(img.shape[:2], bool)
    wig[70:110, 115:135] = True
    result = _include_thin_strands(img, erase, face, head, wig, 100)
    assert (result[face > 0] == 0).all()
    assert (result[head > 0] == 0).all()
    assert (result[wig] == 0).all()


def test_fill_uses_only_background_colours():
    img = np.full((200, 240, 3), 190, np.uint8)
    img[65:145, 145:220] = (0, 0, 255)  # invalid face pixels
    img[130:200, 45:160] = (0, 255, 0)  # invalid jacket pixels
    erase = np.zeros((200, 240), np.uint8)
    erase[75:125, 75:150] = 255
    background = np.ones((200, 240), bool)
    background[65:145, 145:220] = False
    background[130:200, 45:160] = False
    out = fill_from_background(img, erase, background)
    edge = out[80:120, 80:125].astype(np.int16)
    assert np.abs(edge - 190).max() <= 12
    hole = out[88:112, 95:130].reshape(-1, 3).astype(int)
    assert np.abs(hole - 190).max() <= 12


def test_fill_has_no_hard_jumps_inside_large_hole():
    height, width = 300, 360
    gradient = np.tile(np.linspace(80, 180, width, dtype=np.uint8), (height, 1))
    img = np.repeat(gradient[..., None], 3, axis=2)
    erase = np.zeros((height, width), np.uint8)
    erase[80:220, 100:260] = 255
    out = fill_from_background(img, erase, np.ones((height, width), bool))
    interior = out[100:200, 120:240, 0].astype(np.int16)
    max_step = max(
        np.abs(np.diff(interior, axis=0)).max(),
        np.abs(np.diff(interior, axis=1)).max(),
    )
    assert max_step <= 2


def test_fill_without_background_falls_back():
    img = np.full((50, 50, 3), 90, np.uint8)
    erase = np.zeros((50, 50), np.uint8)
    erase[10:20, 10:20] = 255
    out = fill_from_background(img, erase, np.zeros((50, 50), bool))
    assert np.array_equal(out, img)


# ---------- whole fit on the real photo ----------

@needs_photo
@needs_wig
def test_fit_keeps_face_and_eyes_identical():
    img = cv2.imread(str(PHOTO))
    pts, _ = find_landmarks(img)
    out = fit_wig(img, WIGS["side-part"])
    assert out.shape == img.shape
    face = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(face, [pts[FACE_OVAL].astype(np.int32)], 255)
    assert np.array_equal(out[face > 0], img[face > 0])
    for i in [1, 13, 14] + EYES:  # nose, lips, eyes
        x, y = pts[i].astype(int)
        assert (out[y - 3:y + 4, x - 3:x + 4] == img[y - 3:y + 4, x - 3:x + 4]).all(), i
    above = pts[10].astype(int) - [0, 60]  # wig hair above the forehead
    assert not (out[above[1], above[0]] == img[above[1], above[0]]).all()


@needs_photo
@needs_wig
def test_ai_cleanup_only_paints_erased_area():
    img = cv2.imread(str(PHOTO))
    calls = []
    def painter(crop, mask, prompt, seed, negative=None):
        calls.append(prompt)
        return np.zeros_like(crop)
    out = fit_wig(img, WIGS["side-part"], ai_cleanup=True, painter=painter)
    pts, _ = find_landmarks(img)
    x, y = pts[1].astype(int)
    assert len(calls) == 1 and "background" in calls[0]
    assert (out[y, x] == img[y, x]).all()


def test_fit_no_face():
    if not has_image("side-part"):
        pytest.skip("side-part image not generated")
    with pytest.raises(FaceError):
        fit_wig(np.full((300, 300, 3), 255, np.uint8), WIGS["side-part"])


# ---------- API ----------

def test_api_fit_errors(monkeypatch):
    assert post("/fit?wig_id=nope", b"x").status_code == 404
    monkeypatch.setattr(api, "has_image", lambda wid: False)
    assert post("/fit?wig_id=quiff", b"x").status_code == 409
    monkeypatch.setattr(api, "has_image", lambda wid: True)
    assert post("/fit?wig_id=quiff", b"garbage").status_code == 400
    assert post("/fit?wig_id=quiff", b"0" * (10 * 1024 * 1024 + 1)).status_code == 413
    assert client.post("/fit?wig_id=quiff").status_code == 422


@needs_photo
@needs_wig
def test_api_fit_real_photo():
    r = post("/fit?wig_id=side-part", PHOTO.read_bytes())
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"


def test_wig_list_has_gender_and_image_links():
    body = client.get("/wigs").json()
    assert {w["gender"] for w in body} == {"women", "men"}
    for w in body:
        assert w["image"] is None or w["image"] == f"/wig-images/{w['id']}.png"


@needs_photo
def test_recommend_gender_filter():
    r = post("/recommend?top_k=20&gender=men", PHOTO.read_bytes())
    assert r.status_code == 200
    assert {x["wig"]["gender"] for x in r.json()["recommendations"]} == {"men"}
    assert post("/recommend?gender=robots", PHOTO.read_bytes()).status_code == 422
