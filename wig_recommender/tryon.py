"""Virtual try-on: replace the person's hair with the chosen wig using AI.

1. Mask   - find existing hair (MediaPipe hair segmenter) and add the area the
            wig will cover (bigger for long/voluminous wigs). The face is cut
            out of the mask so the AI can never touch it. Only hair changes.
2. Paint  - pre-fill the masked area with the wig colour as rough hair, then the
            local ComfyUI (RealVisXL) refines ONLY that area into realistic hair.
3. Blend  - paste the painted crop back with soft edges. Pixels outside the
            mask (face, background, clothes) are the original photo.
"""
import os
import random
import time
import uuid
from pathlib import Path

import cv2
import httpx
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from .face import CHEEK_L, CHEEK_R, CHIN, FOREHEAD_TOP, FaceError, find_landmarks
from .library import ATTRIBUTES, Wig

COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
CHECKPOINT = os.environ.get("COMFY_CHECKPOINT", "RealVisXL_V5.0_fp16.safetensors")
SEG_MODEL = Path(__file__).resolve().parent.parent / "models" / "selfie_multiclass_256x256.tflite"
HAIR, FACE_SKIN = 1, 3  # class ids in the multiclass segmenter
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
WORK_SIZE = 896  # a bit under SDXL's native 1024: ~40% faster on the M1, still sharp


class TryOnError(RuntimeError):
    """ComfyUI is unreachable or the generation failed."""


# ---------- 1. mask ----------

_segmenter = None


def segment(img: np.ndarray) -> np.ndarray:
    """Per-pixel class: 0 background, 1 hair, 2 body skin, 3 face skin, 4 clothes, 5 other."""
    global _segmenter
    if _segmenter is None:
        if not SEG_MODEL.exists():
            raise TryOnError(f"hair model missing: {SEG_MODEL}")
        _segmenter = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=str(SEG_MODEL)), output_category_mask=True,
            output_confidence_masks=True))
    rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cats = _segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)).category_mask
    return cats.numpy_view().copy()


def segment_confidence(img: np.ndarray, cls: int) -> np.ndarray:
    """0-1 confidence that each pixel belongs to class `cls` (soft edges for cut-outs)."""
    segment(img)  # make sure the segmenter exists
    rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return _segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)).confidence_masks[cls].numpy_view().copy()


# Inner face: brows, eyes, nose, mouth. Never painted, whatever the hair detector says.
CORE = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300, 33, 133, 159, 145, 362, 263, 386, 374,
        1, 4, 98, 327, 61, 291, 0, 17, 152, 148, 377]
NOSE_TIP = 1


def wig_mask(shape: tuple[int, int], pts: np.ndarray, wig: Wig, hair: np.ndarray | None = None,
             skin: np.ndarray | None = None) -> np.ndarray:
    """uint8 mask (255 = AI may paint here) for one wig on one face.

    Only hair changes: the AI may paint (a) the person's current hair and (b) the
    wig's shape OUTSIDE the face. Face skin, eyes, brows, nose and mouth are never
    in the mask, so it cannot add a forehead, eyes or anything else.
    """
    h, w = shape
    face_w = np.linalg.norm(pts[CHEEK_L] - pts[CHEEK_R])
    top_y, chin_y = pts[FOREHEAD_TOP][1], pts[CHIN][1]
    face_h = chin_y - top_y
    cx = (pts[CHEEK_L][0] + pts[CHEEK_R][0]) / 2
    cheek_y = (pts[CHEEK_L][1] + pts[CHEEK_R][1]) / 2
    vol = ATTRIBUTES["volume"].index(wig.volume)  # 0, 1, 2

    half_w = face_w * (0.72 + 0.1 * vol)
    crown_y = top_y - face_h * (0.5 + 0.08 * vol)
    bottom_y = {
        "pixie": cheek_y, "short": chin_y - 0.1 * face_h, "bob": chin_y + 0.2 * face_h,
        "medium": chin_y + 0.45 * face_h, "long": chin_y + 1.1 * face_h,
    }[wig.length]

    m = np.zeros((h, w), np.uint8)
    # dome over the head ...
    cv2.ellipse(m, (int(cx), int(cheek_y)), (int(half_w), int(cheek_y - crown_y)), 0, 180, 360, 255, -1)
    # ... and rounded hair falling down both sides (not over the neck)
    if bottom_y > cheek_y:
        cv2.ellipse(m, (int(cx), int(cheek_y)), (int(half_w), int(bottom_y - cheek_y)), 0, 0, 180, 255, -1)
        if bottom_y > chin_y:
            neck = 0.4 * face_w
            cv2.rectangle(m, (int(cx - neck), int(chin_y)), (int(cx + neck), h), 0, -1)
    # the wig shape never goes over the face itself
    face = np.zeros_like(m)
    cv2.fillPoly(face, [pts[FACE_OVAL].astype(np.int32)], 255)
    m[face > 0] = 0

    if hair is not None:
        # all current hair gets replaced; grow it a little to catch wispy edges.
        # Inside the face outline only hair above the nose counts (fringe, sideburns),
        # so a beard or moustache is left alone.
        k = max(3, int(0.08 * face_w))
        grown = cv2.dilate(hair.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        grown[(face > 0) & ~hair] = 0  # growing must not creep onto facial skin
        grown[int(pts[NOSE_TIP][1]):][face[int(pts[NOSE_TIP][1]):] > 0] = 0
        m[grown > 0] = 255

    if skin is not None:
        m[skin] = 0
    core = np.zeros_like(m)
    cv2.fillConvexPoly(core, cv2.convexHull(pts[CORE].astype(np.int32)), 255)
    m[cv2.dilate(core, np.ones((15, 15), np.uint8)) > 0] = 0
    return m


# ---------- 2. paint with ComfyUI ----------

def wig_prompt(wig: Wig, colour: str | None = None) -> str:
    colour = colour or (wig.colours[0] if wig.colours else "natural")
    fringe = {"none": "no bangs", "side": "side-swept bangs", "full": "full straight-across bangs"}[wig.fringe]
    length = {"pixie": "very short pixie cut", "short": "short", "bob": "chin-length bob",
              "medium": "shoulder-length", "long": "long, past the shoulders"}[wig.length]
    return (f"photo of a person wearing a {wig.name.lower()} wig, {colour} {wig.texture} hair, {length}, "
            f"{fringe}, {wig.volume} volume, realistic individual hair strands, natural hairline, "
            "soft lighting, high detail, photorealistic")


NEGATIVE = ("skin, forehead, face, eyes, eyebrows, bald, hat, cap, helmet, headband, deformed, blurry, "
            "cartoon, painting, text, watermark")

# Wig colour names -> BGR, used to pre-fill the hair area before the AI refines it.
COLOURS = {"black": (25, 22, 20), "dark brown": (30, 40, 60), "brown": (40, 65, 100), "blonde": (110, 175, 215),
           "honey blonde": (70, 140, 190), "auburn": (35, 55, 125), "burgundy": (40, 25, 95)}
DENOISE = 0.8  # < 1 keeps the pre-filled hair shape, so the AI can only refine it into hair


def build_workflow(image_name: str, mask_name: str, prompt: str, seed: int, steps: int = 20,
                   negative: str | None = None) -> dict:
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CHECKPOINT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative or NEGATIVE, "clip": ["1", 1]}},
        "4": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "5": {"class_type": "LoadImageMask", "inputs": {"image": mask_name, "channel": "red"}},
        "6": {"class_type": "InpaintModelConditioning", "inputs": {
            "positive": ["2", 0], "negative": ["3", 0], "vae": ["1", 2],
            "pixels": ["4", 0], "mask": ["5", 0], "noise_mask": True}},
        "7": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["6", 0], "negative": ["6", 1], "latent_image": ["6", 2],
            "seed": seed, "steps": steps, "cfg": 5.5, "sampler_name": "dpmpp_2m",
            "scheduler": "karras", "denoise": DENOISE}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["1", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "wig_tryon"}},
    }


def run_comfy(crop: np.ndarray, mask: np.ndarray, prompt: str, seed: int, negative: str | None = None,
              timeout: float = 600) -> np.ndarray:
    tag = uuid.uuid4().hex[:8]
    try:
        with httpx.Client(base_url=COMFY_URL, timeout=30) as c:
            names = []
            for kind, arr in (("img", crop), ("mask", mask)):
                png = cv2.imencode(".png", arr)[1].tobytes()
                r = c.post("/upload/image", files={"image": (f"wig_{tag}_{kind}.png", png, "image/png")},
                           data={"overwrite": "true"})
                r.raise_for_status()
                names.append(r.json()["name"])
            r = c.post("/prompt", json={"prompt": build_workflow(*names, prompt, seed, negative=negative)})
            if r.status_code != 200:
                raise TryOnError(f"ComfyUI rejected the job: {r.text[:300]}")
            pid = r.json()["prompt_id"]
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                hist = c.get(f"/history/{pid}").json().get(pid)
                if hist:
                    if hist.get("status", {}).get("status_str") == "error":
                        raise TryOnError("ComfyUI generation failed")
                    out = hist["outputs"]["9"]["images"][0]
                    img = c.get("/view", params=out).content
                    return cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
                time.sleep(2)
            raise TryOnError("ComfyUI took too long")
    except httpx.HTTPError as e:
        raise TryOnError(f"cannot reach ComfyUI at {COMFY_URL} ({e.__class__.__name__})") from None


# ---------- 3. crop, blend, and the whole pipeline ----------

def head_crop(mask: np.ndarray, pad: float = 0.15) -> tuple[int, int, int, int]:
    """Square-ish box around the mask, clipped to the image: x0, y0, x1, y1."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise FaceError("nothing to paint")
    h, w = mask.shape
    side = max(xs.max() - xs.min(), ys.max() - ys.min()) * (1 + 2 * pad)
    cx, cy = (xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2
    x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    return x0, y0, int(min(w, cx + side / 2)), int(min(h, cy + side / 2))


def _work_size(cw: int, ch: int) -> tuple[int, int]:
    s = WORK_SIZE / max(cw, ch)
    return max(64, round(cw * s / 64) * 64), max(64, round(ch * s / 64) * 64)


def prefill(crop: np.ndarray, mask: np.ndarray, wig: Wig, colour: str | None = None) -> np.ndarray:
    """Paint the masked area as rough hair (wig colour + vertical strand texture).
    The AI then only has to turn this into realistic hair; it starts from hair, not
    from whatever was there, so it does not invent skin, a forehead or eyes."""
    name = (colour or (wig.colours[0] if wig.colours else "brown")).lower()
    base = np.array(COLOURS.get(name, COLOURS["brown"]), np.float32)
    rng = np.random.default_rng(0)
    h, w = mask.shape
    strands = cv2.blur(rng.normal(0, 1, (h, w)).astype(np.float32), (1, max(3, h // 25)))
    tex = base + 18 * strands[..., None]
    out = crop.astype(np.float32)
    out[mask > 0] = tex[mask > 0]
    return np.clip(out, 0, 255).astype(np.uint8)


def try_on(img: np.ndarray, wig: Wig, colour: str | None = None, seed: int | None = None,
           painter=run_comfy) -> tuple[np.ndarray, np.ndarray]:
    """Returns (result image, mask used). `painter` is swappable for tests."""
    pts, _ = find_landmarks(img)
    cats = segment(img)
    mask = wig_mask(img.shape[:2], pts, wig, cats == HAIR, cats == FACE_SKIN)
    x0, y0, x1, y1 = head_crop(mask)
    crop, crop_mask = img[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    ww, wh = _work_size(x1 - x0, y1 - y0)
    grown = cv2.dilate(crop_mask, np.ones((9, 9), np.uint8))  # paint a little past the edge
    painted = painter(cv2.resize(prefill(crop, grown, wig, colour), (ww, wh), interpolation=cv2.INTER_AREA),
                      cv2.resize(grown, (ww, wh), interpolation=cv2.INTER_NEAREST),
                      wig_prompt(wig, colour), seed if seed is not None else random.randrange(2**32))
    painted = cv2.resize(painted, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LANCZOS4)

    feather = max(3, int(0.04 * (x1 - x0)) | 1)
    alpha = cv2.GaussianBlur(crop_mask, (feather, feather), 0)[..., None] / 255.0
    out = img.copy()
    out[y0:y1, x0:x1] = (alpha * painted + (1 - alpha) * crop).astype(np.uint8)
    return out, mask
