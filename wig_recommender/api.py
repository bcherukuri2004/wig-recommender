"""HTTP API so the avatar app (or Aldrin's flow) can call the recommender.

Run:  .venv/bin/uvicorn wig_recommender.api:app --port 8100
"""
import os
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import cv2
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .face import MAX_BYTES, FaceError, decode_image
from .fit import WIG_DIR, fit_wig, has_image
from .library import DEFAULT_PATH, load_library
from .recommend import recommend
from .tryon import TryOnError, try_on

app = FastAPI(title="Wig Recommender")
WIGS = load_library(os.environ.get("WIG_LIBRARY", DEFAULT_PATH))
WIG_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/wig-images", StaticFiles(directory=WIG_DIR), name="wig-images")


def _wig_json(wig):
    d = {k: v for k, v in asdict(wig).items() if k != "suitability"}
    d["image"] = f"/wig-images/{wig.id}.png" if has_image(wig.id) else None
    return d


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "wigs": len(WIGS)}


@app.get("/wigs")
def list_wigs():
    return [_wig_json(w) for w in WIGS.values()]


@app.get("/wigs/{wig_id}")
def get_wig(wig_id: str):
    if wig_id not in WIGS:
        raise HTTPException(404, f"unknown wig id {wig_id!r}")
    return _wig_json(WIGS[wig_id])


@app.post("/recommend")
async def recommend_endpoint(
    photo: UploadFile = File(...),
    top_k: int = Query(3, ge=1, le=20),
    length: Literal["pixie", "short", "bob", "medium", "long"] | None = None,
    texture: Literal["straight", "wavy", "curly"] | None = None,
    colour: str | None = Query(None, max_length=40),
    gender: Literal["women", "men"] | None = None,
):
    data = await photo.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "image larger than 10 MB")
    try:
        face, probs, ranked = recommend(data, WIGS, top_k=top_k, length=length,
                                        texture=texture, colour=colour, gender=gender)
    except FaceError as e:
        raise _face_error(e) from None
    return {
        "face": {
            "shape": max(probs, key=lambda s: probs[s]),
            "probabilities": probs,
            "measurements": {k: round(getattr(face, k), 3) for k in ("length", "forehead", "jaw", "chin")},
            "box": face.box,
            "warnings": face.warnings,
        },
        "recommendations": [
            {"wig": _wig_json(r["wig"]), "score": r["score"], "reasons": r["reasons"]} for r in ranked
        ],
    }


def _face_error(e: FaceError) -> HTTPException:
    return HTTPException(400 if "image" in str(e) or "empty" in str(e) else 422, str(e))


@app.post("/tryon")
def tryon_endpoint(
    photo: UploadFile = File(...),
    wig_id: str = Query(..., max_length=100),
    colour: str | None = Query(None, max_length=40),
    seed: int | None = Query(None, ge=0, lt=2**32),
):
    """Replace the hair with the chosen wig (local ComfyUI, ~3 min on an M1). Returns a JPEG."""
    if wig_id not in WIGS:
        raise HTTPException(404, f"unknown wig id {wig_id!r}")
    data = photo.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "image larger than 10 MB")
    try:
        out, _ = try_on(decode_image(data), WIGS[wig_id], colour=colour, seed=seed)
    except FaceError as e:
        raise _face_error(e) from None
    except TryOnError as e:
        raise HTTPException(503, str(e)) from None
    return Response(cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes(), media_type="image/jpeg")


@app.post("/fit")
def fit_endpoint(
    photo: UploadFile = File(...),
    wig_id: str = Query(..., max_length=100),
    ai: bool = False,
):
    """Place a catalogue wig and fill exposed hair regions from the background.
    ai=true also asks the local AI to clean up erased regions."""
    if wig_id not in WIGS:
        raise HTTPException(404, f"unknown wig id {wig_id!r}")
    if not has_image(wig_id):
        raise HTTPException(409, f"wig {wig_id!r} has no catalogue image yet")
    data = photo.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "image larger than 10 MB")
    try:
        out = fit_wig(decode_image(data), WIGS[wig_id], ai_cleanup=ai)
    except FaceError as e:
        raise _face_error(e) from None
    except TryOnError as e:
        raise HTTPException(503, str(e)) from None
    return Response(cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes(), media_type="image/jpeg")
