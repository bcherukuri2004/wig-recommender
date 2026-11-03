"""Step 1: find the face and measure its proportions.

Uses MediaPipe Face Mesh (468 landmarks, runs on CPU, model ships inside the
pip wheel). We only keep a handful of ratios, all divided by cheekbone width
so the numbers don't depend on photo size or how close the camera was.
"""
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np

MAX_BYTES = 10 * 1024 * 1024
MIN_FACE_PX = 80  # cheek-to-cheek width below this is too small to measure

# MediaPipe Face Mesh landmark ids
FOREHEAD_TOP, CHIN = 10, 152
CHEEK_L, CHEEK_R = 234, 454
TEMPLE_L, TEMPLE_R = 54, 284
JAW_L, JAW_R = 172, 397
CHIN_L, CHIN_R = 176, 400
NOSE_TIP = 1


class FaceError(ValueError):
    """Raised for input we can't analyse (bad image, no face, face too small)."""


@dataclass
class FaceMeasurements:
    length: float          # forehead-to-chin / cheek width
    forehead: float        # temple width / cheek width
    jaw: float             # jaw-corner width / cheek width
    chin: float            # chin width / cheek width
    cheek_px: float        # raw cheek width in pixels (for size checks)
    box: tuple[int, int, int, int]  # x, y, w, h of the face in the image
    warnings: list[str] = field(default_factory=list)

    def vector(self) -> np.ndarray:
        return np.array([self.length, self.forehead, self.jaw, self.chin])


def decode_image(data: bytes) -> np.ndarray:
    if not data:
        raise FaceError("empty upload")
    if len(data) > MAX_BYTES:
        raise FaceError(f"image larger than {MAX_BYTES // (1024 * 1024)} MB")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FaceError("file is not a readable image")
    return img


_mesh = None


def _get_mesh():
    global _mesh
    if _mesh is None:
        _mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=3)
    return _mesh


def find_landmarks(img: np.ndarray) -> tuple[np.ndarray, int]:
    """Pixel (x, y) of the 468 points on the largest face, plus how many faces were seen."""
    h, w = img.shape[:2]
    result = _get_mesh().process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not result.multi_face_landmarks:
        raise FaceError("no face found in the photo")
    faces = [np.array([(p.x * w, p.y * h) for p in f.landmark]) for f in result.multi_face_landmarks]
    width = lambda pts: np.linalg.norm(pts[CHEEK_L] - pts[CHEEK_R])
    return max(faces, key=width), len(faces)  # biggest face = the person in front


def measure_face(img: np.ndarray) -> FaceMeasurements:
    pts, n_faces = find_landmarks(img)
    warnings = []
    if n_faces > 1:
        warnings.append(f"{n_faces} faces found; using the largest")

    d = lambda a, b: float(np.linalg.norm(pts[a] - pts[b]))
    cheek = d(CHEEK_L, CHEEK_R)
    if cheek < MIN_FACE_PX:
        raise FaceError("face too small; use a closer photo")

    # Head turned sideways makes the face look narrower -> wrong shape.
    centre_x = (pts[CHEEK_L][0] + pts[CHEEK_R][0]) / 2
    if abs(pts[NOSE_TIP][0] - centre_x) / cheek > 0.12:
        warnings.append("face is turned; a straight-on photo gives a better match")

    x0, y0 = pts.min(axis=0).astype(int)
    x1, y1 = pts.max(axis=0).astype(int)
    return FaceMeasurements(
        length=d(FOREHEAD_TOP, CHIN) / cheek,
        forehead=d(TEMPLE_L, TEMPLE_R) / cheek,
        jaw=d(JAW_L, JAW_R) / cheek,
        chin=d(CHIN_L, CHIN_R) / cheek,
        cheek_px=cheek,
        box=(int(x0), int(y0), int(x1 - x0), int(y1 - y0)),
        warnings=warnings,
    )
