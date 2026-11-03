# Wig Recommender

Upload a face photo and get the best-fitting wigs from a preset library.

1. **Find the face**: MediaPipe Face Mesh (468 points, CPU, runs on a Mac) measures face length, forehead, jaw and chin width relative to cheekbone width.
2. **Face shape**: the measurements are compared to six typical shapes (oval, round, square, oblong, heart, diamond), which gives a probability for each.
3. **Best match**: each wig (`data/wigs.json`) is scored by stylist rules for each shape. The final score is the probability-weighted sum. A wig can carry its own `suitability` scores to override the rules.

4. **Try on (AI)**: `POST /tryon?wig_id=...` swaps the person's hair for the wig. MediaPipe's hair segmenter (`models/selfie_multiclass_256x256.tflite`, Google, ~16 MB) finds the current hair. A mask is built over the hair and the wig area, leaving the face out. RealVisXL repaints only the masked area through the local ComfyUI (`COMFY_URL`, default `127.0.0.1:8188`). The face and background stay the original photo. Takes 3–10 minutes per image on an M1 Pro.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q                       # tests
.venv/bin/python -m wig_recommender.cli photo.jpg   # command line
.venv/bin/uvicorn wig_recommender.api:app --port 8100  # API: POST /recommend (form field "photo")
```

API errors (both endpoints): 400 for an empty or non-image file, 413 for files over 10 MB, 422 when no face is found or the face is too small, 404 for an unknown wig id, 503 when ComfyUI is down.
