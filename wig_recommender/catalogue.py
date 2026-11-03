"""Build the catalogue images: one front-facing, wig-only PNG per wig.

For each wig we ask the local ComfyUI for a front-facing studio portrait of a
person wearing it, then:
  - cut out ONLY the hair (MediaPipe hair segmenter -> transparent background)
  - record where the model's face points were (temples, forehead, chin), so the
    try-on can line the wig up with a customer's face later.

Run:  .venv/bin/python -m wig_recommender.catalogue [wig_id ...]
Output: data/wigs/<id>.png (wig only), data/wigs/<id>.json (anchors),
        data/wigs/src/<id>.jpg (the generated portrait, for reference)
"""
import json
import sys
import time

import cv2
import httpx
import numpy as np

from .face import CHEEK_L, CHEEK_R, CHIN, FOREHEAD_TOP, FaceError, find_landmarks
from .library import DEFAULT_PATH, Wig, load_library
from .tryon import CHECKPOINT, COMFY_URL, HAIR, segment_confidence

OUT = DEFAULT_PATH.parent / "wigs"
ANCHORS = {"cheek_l": CHEEK_L, "cheek_r": CHEEK_R, "forehead": FOREHEAD_TOP, "chin": CHIN}
LENGTH = {"pixie": "very short", "short": "short", "bob": "chin-length bob", "medium": "shoulder-length",
          "long": "long, past the shoulders"}
FRINGE = {"none": "no bangs", "side": "side-swept fringe", "full": "full fringe"}


def portrait_prompt(wig: Wig) -> str:
    person = {"men": "man", "women": "woman"}.get(wig.gender, "person")
    colour = wig.colours[0] if wig.colours else "brown"
    return (f"studio portrait photo of a {person} with a {wig.name.lower()} hairstyle, {colour} {wig.texture} hair, "
            f"{LENGTH[wig.length]}, {FRINGE[wig.fringe]}, {wig.volume} volume, front view, facing the camera, "
            "head and shoulders, neutral expression, plain light grey background, soft even lighting, "
            "sharp focus, photorealistic")


NEGATIVE = "hat, hands, side view, profile, tilted head, multiple people, text, watermark, blurry, cartoon"


def generate(prompt: str, seed: int, w: int = 640, h: int = 832, timeout: float = 3600) -> np.ndarray:
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CHECKPOINT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0],
            "seed": seed, "steps": 16, "cfg": 5.0, "sampler_name": "dpmpp_2m", "scheduler": "karras",
            "denoise": 1.0}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "wig_catalogue"}},
    }
    with httpx.Client(base_url=COMFY_URL, timeout=30) as c:
        pid = c.post("/prompt", json={"prompt": wf}).raise_for_status().json()["prompt_id"]
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            hist = c.get(f"/history/{pid}").json().get(pid)
            if hist:
                out = hist["outputs"]["7"]["images"][0]
                return cv2.imdecode(np.frombuffer(c.get("/view", params=out).content, np.uint8), cv2.IMREAD_COLOR)
            time.sleep(3)
    raise TimeoutError("ComfyUI took too long")


def cut_out_wig(img: np.ndarray) -> tuple[np.ndarray, dict]:
    """Returns (RGBA wig-only image cropped to the hair, anchor points in that crop)."""
    pts, _ = find_landmarks(img)
    alpha = segment_confidence(img, HAIR)
    hard = (alpha > 0.5).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(hard)
    if n < 2:
        raise FaceError("no hair found")
    keep = 1 + np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1]
    big = np.isin(labels, [k for k in keep if stats[k, cv2.CC_STAT_AREA] > 0.05 * stats[keep[0], cv2.CC_STAT_AREA]])
    alpha = np.where(cv2.dilate(big.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0, alpha, 0)
    if (alpha > 0.5).sum() < 0.02 * alpha.size:
        raise FaceError("too little hair")
    ys, xs = np.nonzero(alpha > 0.05)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    rgba = np.dstack([img, (alpha * 255).astype(np.uint8)])[y0:y1, x0:x1]
    anchors = {k: [float(pts[i][0] - x0), float(pts[i][1] - y0)] for k, i in ANCHORS.items()}
    return rgba, anchors


def build(wig: Wig, tries: int = 3) -> None:
    (OUT / "src").mkdir(parents=True, exist_ok=True)
    for attempt in range(tries):
        t0 = time.monotonic()
        img = generate(portrait_prompt(wig), seed=1000 + attempt)
        try:
            rgba, anchors = cut_out_wig(img)
        except FaceError as e:
            print(f"{wig.id}: attempt {attempt + 1} unusable ({e}), retrying", flush=True)
            continue
        cv2.imwrite(str(OUT / "src" / f"{wig.id}.jpg"), img)
        cv2.imwrite(str(OUT / f"{wig.id}.png"), rgba)
        (OUT / f"{wig.id}.json").write_text(json.dumps({"anchors": anchors, "size": rgba.shape[1::-1]}))
        print(f"{wig.id}: done in {time.monotonic() - t0:.0f}s", flush=True)
        return
    print(f"{wig.id}: FAILED after {tries} attempts", flush=True)


if __name__ == "__main__":
    wigs = load_library()
    for wid in sys.argv[1:] or list(wigs):
        if (OUT / f"{wid}.png").exists():
            print(f"{wid}: already done, skipping", flush=True)
            continue
        try:
            build(wigs[wid])
        except Exception as e:  # one slow/failed wig must not stop the rest
            print(f"{wid}: FAILED ({e.__class__.__name__}: {e})", flush=True)
