"""Step 2: turn face measurements into face-shape probabilities.

Each shape has a "typical" set of ratios (a prototype). We score how close the
face is to each prototype and convert that to probabilities, so a face that is
between oval and round gets e.g. 55% / 40% instead of a hard yes/no.
Prototype numbers are hand-set from stylist definitions and calibrated on a
MediaPipe measurement; they are heuristics, not a trained model.
"""
import numpy as np

from .face import FaceMeasurements

SHAPES = ["oval", "round", "square", "oblong", "heart", "diamond"]

#                      length forehead jaw   chin
PROTOTYPES = np.array([
    [1.15, 0.85, 0.80, 0.36],  # oval: a bit longer than wide, balanced
    [1.02, 0.86, 0.86, 0.40],  # round: short, soft wide jaw
    [1.08, 0.90, 0.90, 0.44],  # square: wide forehead and wide jaw
    [1.30, 0.86, 0.82, 0.38],  # oblong: clearly long
    [1.13, 0.92, 0.76, 0.31],  # heart: wide forehead, narrow pointed chin
    [1.15, 0.78, 0.76, 0.33],  # diamond: cheekbones widest, narrow top and bottom
])
SPREAD = np.array([0.06, 0.04, 0.04, 0.04])  # how much each ratio normally varies


def shape_probabilities(m: FaceMeasurements) -> dict[str, float]:
    z = (m.vector() - PROTOTYPES) / SPREAD
    logits = -0.5 * (z ** 2).sum(axis=1)
    p = np.exp(logits - logits.max())
    p /= p.sum()
    return {s: round(float(v), 4) for s, v in zip(SHAPES, p)}
