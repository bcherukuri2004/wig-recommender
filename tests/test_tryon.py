from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import wig_recommender.api as api
from wig_recommender.face import FaceError, find_landmarks
from wig_recommender.library import load_library
from wig_recommender.tryon import (CHIN, FOREHEAD_TOP, TryOnError, build_workflow, head_crop, prefill,
                                   run_comfy, try_on, wig_mask, wig_prompt)

PHOTO = Path.home() / "ai-avatar-platform/test-photo/bhargav.jpg"
needs_photo = pytest.mark.skipif(not PHOTO.exists(), reason="test photo missing")
WIGS = load_library()
client = TestClient(api.app)


@pytest.fixture(scope="module")
def face():
    img = cv2.imread(str(PHOTO))
    return img, find_landmarks(img)[0]


def red_painter(crop, mask, prompt, seed):
    assert crop.shape[:2] == mask.shape[:2] and crop.shape[0] % 64 == 0 and crop.shape[1] % 64 == 0
    return np.full_like(crop, (0, 0, 255))


# ---------- mask ----------

@needs_photo
def test_mask_never_touches_eyes_nose_mouth(face):
    img, pts = face
    for wig in WIGS.values():
        m = wig_mask(img.shape[:2], pts, wig, np.ones(img.shape[:2], bool))  # even if "all hair"
        for i in (1, 4, 13, 14, 33, 263, 152):  # nose, mouth, eye corners, chin
            x, y = pts[i].astype(int)
            assert m[y, x] == 0, (wig.id, i)


@needs_photo
def test_mask_covers_crown_and_longer_wigs_reach_lower(face):
    img, pts = face
    above = pts[FOREHEAD_TOP].astype(int) - [0, 40]
    lowest = {}
    for wid in ("pixie-sleek", "quiff", "long-waves"):
        m = wig_mask(img.shape[:2], pts, WIGS[wid])
        assert m[above[1], above[0]] == 255
        lowest[wid] = np.nonzero(m)[0].max()
    assert lowest["pixie-sleek"] < lowest["quiff"] < lowest["long-waves"]


@needs_photo
def test_forehead_never_painted_even_for_fringe_wigs(face):
    img, pts = face
    forehead = (pts[FOREHEAD_TOP] * 0.97 + pts[CHIN] * 0.03).astype(int)  # between top landmark and brows
    for wid in ("bob-fringe", "quiff", "curly-top"):
        assert wig_mask(img.shape[:2], pts, WIGS[wid])[forehead[1], forehead[0]] == 0


@needs_photo
def test_hair_on_forehead_replaced_but_beard_kept(face):
    img, pts = face
    hair = np.ones(img.shape[:2], bool)  # pretend hair everywhere, beard included
    m = wig_mask(img.shape[:2], pts, WIGS["quiff"], hair)
    fx, fy = (pts[FOREHEAD_TOP] * 0.97 + pts[CHIN] * 0.03).astype(int)
    assert m[fy, fx] == 255  # hair hanging over the forehead -> becomes wig
    for i in (152, 148, 377, 61, 291):  # chin and mouth corners (beard / moustache area)
        x, y = pts[i].astype(int)
        assert m[y, x] == 0, i


@needs_photo
def test_face_skin_is_never_painted(face):
    img, pts = face
    skin = np.zeros(img.shape[:2], bool)
    skin[: img.shape[0] // 2] = True  # pretend the whole top half is face skin
    m = wig_mask(img.shape[:2], pts, WIGS["long-waves"], np.ones_like(skin), skin)
    assert not m[skin].any()


def test_prefill_only_changes_masked_pixels():
    crop = np.full((64, 64, 3), 200, np.uint8)
    mask = np.zeros((64, 64), np.uint8)
    mask[:32] = 255
    out = prefill(crop, mask, WIGS["quiff"])  # black wig
    assert (out[32:] == 200).all()
    assert out[:32].mean() < 80


def test_prefill_unknown_colour_falls_back():
    out = prefill(np.zeros((8, 8, 3), np.uint8), np.full((8, 8), 255, np.uint8), WIGS["quiff"], "neon green")
    assert out.shape == (8, 8, 3)


def test_head_crop_empty_mask():
    with pytest.raises(FaceError):
        head_crop(np.zeros((100, 100), np.uint8))


def test_head_crop_stays_inside_image():
    m = np.zeros((100, 200), np.uint8)
    m[0:50, 150:200] = 255  # touching the corner
    x0, y0, x1, y1 = head_crop(m)
    assert 0 <= x0 < x1 <= 200 and 0 <= y0 < y1 <= 100


# ---------- prompt / workflow ----------

def test_prompt_uses_wig_and_colour_override():
    p = wig_prompt(WIGS["bob-fringe"], colour="burgundy")
    assert "burgundy" in p and "bangs" in p and "bob" in p
    assert "black" in wig_prompt(WIGS["bob-fringe"])  # first catalogue colour by default


def test_workflow_is_wired_correctly():
    wf = build_workflow("a.png", "b.png", "prompt", seed=7)
    for node in wf.values():  # every link points at a node that exists
        for v in node["inputs"].values():
            if isinstance(v, list):
                assert v[0] in wf
    assert wf["7"]["inputs"]["seed"] == 7 and wf["5"]["inputs"]["image"] == "b.png"
    assert 0 < wf["7"]["inputs"]["denoise"] < 1  # refine the pre-filled hair, don't repaint from scratch


def test_comfy_unreachable(monkeypatch):
    monkeypatch.setattr("wig_recommender.tryon.COMFY_URL", "http://127.0.0.1:9")
    with pytest.raises(TryOnError, match="cannot reach"):
        run_comfy(np.zeros((64, 64, 3), np.uint8), np.zeros((64, 64), np.uint8), "x", 1)


# ---------- whole pipeline with a fake painter ----------

@needs_photo
def test_try_on_keeps_face_pixels_identical(face):
    img, pts = face
    out, mask = try_on(img, WIGS["lob-layered"], seed=1, painter=red_painter)
    assert out.shape == img.shape
    x, y = pts[1].astype(int)  # nose tip
    assert (out[y - 20:y + 20, x - 20:x + 20] == img[y - 20:y + 20, x - 20:x + 20]).all()
    hy, hx = np.argwhere(cv2.erode(mask, np.ones((25, 25), np.uint8)) > 0)[0]
    assert np.abs(out[hy, hx].astype(int) - (0, 0, 255)).max() <= 3  # painted (allowing resize rounding)


def test_try_on_no_face():
    with pytest.raises(FaceError):
        try_on(np.zeros((300, 300, 3), np.uint8), WIGS["lob-layered"], painter=red_painter)


# ---------- API ----------

def post(url, data=b"x"):
    return client.post(url, files={"photo": ("p.jpg", data, "image/jpeg")})


@pytest.mark.parametrize("url,code", [
    ("/tryon", 422),                         # wig_id missing
    ("/tryon?wig_id=nope", 404),
    ("/tryon?wig_id=" + "x" * 101, 422),
    ("/tryon?wig_id=quiff&seed=-1", 422),
    ("/tryon?wig_id=quiff", 400),        # b"x" is not an image
])
def test_tryon_bad_requests(url, code):
    assert post(url).status_code == code


def test_tryon_oversized():
    assert post("/tryon?wig_id=quiff", b"0" * (10 * 1024 * 1024 + 1)).status_code == 413


@needs_photo
def test_tryon_api_with_fake_painter(monkeypatch):
    monkeypatch.setattr(api, "try_on", lambda img, wig, **kw: try_on(img, wig, painter=red_painter, **kw))
    r = post("/tryon?wig_id=lob-layered&seed=3", PHOTO.read_bytes())
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR).shape == cv2.imread(str(PHOTO)).shape


@needs_photo
def test_tryon_api_comfy_down_is_503(monkeypatch):
    monkeypatch.setattr("wig_recommender.tryon.COMFY_URL", "http://127.0.0.1:9")
    assert post("/tryon?wig_id=lob-layered", PHOTO.read_bytes()).status_code == 503

