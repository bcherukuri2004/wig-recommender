"""Lenskart-style try-on: put the catalogue wig image itself on the customer's photo.

1. Line up  - each wig PNG remembers where the model's temples, forehead and
              chin were. We find the same points on the customer and compute one
              scale + rotation + shift that maps one set onto the other.
2. Erase    - the customer's own hair that would stick out from under the wig is
              filled from nearby background pixels.
3. Place    - the wig image is laid on top, outside the face. The eyes stay visible.
The customer's face is never inpainted or covered.
"""
import json
from pathlib import Path

import cv2
import numpy as np

from .face import CHEEK_L, CHEEK_R, FaceError, find_landmarks
from .library import DEFAULT_PATH, Wig
from .tryon import CORE, FACE_OVAL, HAIR, run_comfy, segment, segment_confidence

WIG_DIR = DEFAULT_PATH.parent / "wigs"
ANCHOR_IDS = {"cheek_l": 234, "cheek_r": 454, "forehead": 10, "chin": 152}
EYES = [33, 133, 159, 145, 362, 263, 386, 374]
ERASE_PROMPT = "empty plain background, smooth wall, bare neck and shoulders, photorealistic, sharp"
ERASE_NEGATIVE = "hair, curls, strands, wisps, fog, haze, glow, face, eyes, person, text, watermark"


def wig_image_path(wig_id: str) -> Path:
    return WIG_DIR / f"{wig_id}.png"


def has_image(wig_id: str) -> bool:
    return wig_image_path(wig_id).exists() and (WIG_DIR / f"{wig_id}.json").exists()


def load_wig_image(wig_id: str) -> tuple[np.ndarray, dict]:
    if not has_image(wig_id):
        raise FileNotFoundError(f"no catalogue image for {wig_id!r}")
    rgba = cv2.imread(str(wig_image_path(wig_id)), cv2.IMREAD_UNCHANGED)
    meta = json.loads((WIG_DIR / f"{wig_id}.json").read_text())
    return rgba, meta["anchors"]


def placement(anchors: dict, pts: np.ndarray) -> np.ndarray:
    """2x3 matrix (scale + rotation + shift) taking wig-image points onto the face."""
    src = np.array([anchors[k] for k in ANCHOR_IDS], np.float32)
    dst = np.array([pts[i] for i in ANCHOR_IDS.values()], np.float32)
    m, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if m is None:
        raise FaceError("could not line the wig up with the face")
    return m


def _poly(shape, pts, ids, convex=False) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    p = pts[ids].astype(np.int32)
    if convex:
        cv2.fillConvexPoly(m, cv2.convexHull(p), 255)
    else:
        cv2.fillPoly(m, [p], 255)
    return m


def fit_wig(img: np.ndarray, wig: Wig, ai_cleanup: bool = False, painter=run_comfy,
            seed: int = 1, blackhat_threshold: int = 12) -> np.ndarray:
    pts, _ = find_landmarks(img)
    rgba, anchors = load_wig_image(wig.id)
    h, w = img.shape[:2]

    # 1. line up the wig image with this face
    m = placement(anchors, pts)
    wig_rgb = cv2.warpAffine(rgba[..., :3], m, (w, h), flags=cv2.INTER_LINEAR)
    alpha = cv2.warpAffine(rgba[..., 3], m, (w, h), flags=cv2.INTER_LINEAR).astype(np.float32) / 255
    wig_area = alpha > 0.05
    eyes = cv2.dilate(_poly((h, w), pts, EYES, convex=True), np.ones((21, 21), np.uint8))
    alpha[eyes > 0] = 0  # a long fringe never hides the eyes

    # 2. erase the customer's hair that the wig doesn't cover (never the face)
    face = _poly((h, w), pts, FACE_OVAL) | _poly((h, w), pts, CORE, convex=True)
    alpha[face > 0] = 0  # never blend catalogue pixels over the customer's face
    # low threshold + grow so faint wispy curls are caught too
    hair = (segment_confidence(img, HAIR) > 0.1).astype(np.uint8) * 255
    hair = cv2.dilate(hair, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    # Inside the head outline (face + wig) hair is never erased: erasing there would
    # expose a "new forehead". Only hair sticking out beyond the head is replaced.
    head_pts = np.vstack([pts[FACE_OVAL], np.argwhere(alpha > 0.3)[:, ::-1]]).astype(np.int32)
    head = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(head, cv2.convexHull(head_pts), 255)
    erase = ((hair > 0) & (alpha < 0.6) & (face == 0) & (head == 0)).astype(np.uint8) * 255
    erase = _include_thin_strands(
        img, erase, face, head, wig_area,
        float(np.linalg.norm(pts[CHEEK_L] - pts[CHEEK_R])), blackhat_threshold,
    )
    base = img
    if erase.any():
        base = fill_from_background(img, erase, segment(img) == 0)
        if ai_cleanup:
            base = _ai_erase(base, erase, painter, seed)

    # 3. lay the wig on top
    blended_alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    blended_alpha[face > 0] = 0
    a = blended_alpha[..., None]
    out = (a * wig_rgb + (1 - a) * base).astype(np.uint8)
    out[face > 0] = img[face > 0]
    return out


def _include_thin_strands(
    img: np.ndarray,
    erase: np.ndarray,
    face: np.ndarray,
    head: np.ndarray,
    wig_area: np.ndarray,
    cheek_width: float,
    blackhat_threshold: int = 20,
) -> np.ndarray:
    """Add nearby thin dark strands without extending into face, head, or wig."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blackhat = cv2.morphologyEx(
        gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    )
    band_size = max(3, int(round(0.6 * cheek_width)) | 1)
    band = cv2.dilate(
        (erase > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_size, band_size)),
    ) > 0
    allowed = band & (erase == 0) & (face == 0) & (head == 0) & ~wig_area
    dark = (gray < 120) & allowed

    count, labels, _, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8), 8)
    if count <= 1:
        return erase
    distance = cv2.distanceTransform(dark.astype(np.uint8), cv2.DIST_L2, 5)
    max_distance = np.zeros(count, np.float32)
    np.maximum.at(max_distance, labels.ravel(), distance.ravel())
    thin_labels = np.flatnonzero((max_distance <= 7.5) & (np.arange(count) > 0))
    thin_dark = np.isin(labels, thin_labels)
    strands = (blackhat > blackhat_threshold) & thin_dark
    strands = cv2.dilate(strands.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    strands &= allowed
    return erase | (strands.astype(np.uint8) * 255)


def fill_from_background(img: np.ndarray, erase: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Push valid background colours into erased areas through a weighted pyramid."""
    valid = (background & (erase == 0)).astype(np.float32)
    if not valid.any():
        return img.copy()

    image_levels = [img.astype(np.float32)]
    weight_levels = [valid]
    while min(weight_levels[-1].shape) > 1 and not np.all(weight_levels[-1] > 1e-6):
        current_image, current_weight = image_levels[-1], weight_levels[-1]
        numerator = cv2.pyrDown(current_image * current_weight[..., None])
        weight = cv2.pyrDown(current_weight)
        image = np.zeros_like(numerator)
        covered = weight > 1e-6
        image[covered] = numerator[covered] / weight[covered, None]
        image_levels.append(image)
        weight_levels.append(weight)

    fill = image_levels[-1]
    for level in range(len(image_levels) - 2, -1, -1):
        height, width = weight_levels[level].shape
        fill = cv2.pyrUp(fill, dstsize=(width, height))
        real = weight_levels[level] > 1e-6
        fill[real] = image_levels[level][real]

    out = img.copy()
    mask = erase > 0
    out[mask] = np.clip(fill[mask], 0, 255).astype(np.uint8)
    soft = cv2.GaussianBlur(erase, (5, 5), 0)[..., None] / 255.0
    return (soft * out + (1 - soft) * img).astype(np.uint8)


def _ai_erase(base: np.ndarray, erase: np.ndarray, painter, seed: int) -> np.ndarray:
    """Let the AI turn the rough fill into clean background. Only the erased area changes."""
    ys, xs = np.nonzero(erase)
    pad = 64
    x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
    x1, y1 = min(base.shape[1], xs.max() + pad), min(base.shape[0], ys.max() + pad)
    crop, mask = base[y0:y1, x0:x1], erase[y0:y1, x0:x1]
    s = 896 / max(crop.shape[:2])
    size = (max(64, round(crop.shape[1] * s / 64) * 64), max(64, round(crop.shape[0] * s / 64) * 64))
    painted = painter(cv2.resize(crop, size, interpolation=cv2.INTER_AREA),
                      cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST), ERASE_PROMPT, seed,
                      negative=ERASE_NEGATIVE)
    painted = cv2.resize(painted, crop.shape[1::-1], interpolation=cv2.INTER_LANCZOS4)
    a = cv2.GaussianBlur(mask, (9, 9), 0)[..., None] / 255.0
    out = base.copy()
    out[y0:y1, x0:x1] = (a * painted + (1 - a) * crop).astype(np.uint8)
    return out
