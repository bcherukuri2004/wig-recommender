from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import wig_recommender.api as api
from wig_recommender.face import FaceError, find_landmarks
from wig_recommender.fit import EYES, fill_from_background, fit_wig, has_image, placement
from wig_recommender.library import load_library

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


# ---------- background fill ----------

def test_fill_uses_only_background_colours():
    img = np.zeros((200, 200, 3), np.uint8)
    img[:, :] = (200, 200, 200)        # grey wall
    img[80:120, 80:120] = (0, 0, 255)  # red "face" right next to the hole
    erase = np.zeros((200, 200), np.uint8)
    erase[60:80, 60:140] = 255
    bg = np.ones((200, 200), bool)
    bg[80:120, 80:120] = False
    out = fill_from_background(img, erase, bg)
    hole = out[65:75, 70:130].reshape(-1, 3).astype(int)
    assert np.abs(hole - 200).max() < 15  # filled with wall grey, no red dragged in


def test_fill_without_background_falls_back():
    img = np.full((50, 50, 3), 90, np.uint8)
    erase = np.zeros((50, 50), np.uint8)
    erase[10:20, 10:20] = 255
    assert fill_from_background(img, erase, np.zeros((50, 50), bool)).shape == img.shape


# ---------- whole fit on the real photo ----------

@needs_photo
@needs_wig
def test_fit_keeps_face_and_eyes_identical():
    img = cv2.imread(str(PHOTO))
    pts, _ = find_landmarks(img)
    out = fit_wig(img, WIGS["side-part"])
    assert out.shape == img.shape
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
