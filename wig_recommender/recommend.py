"""Step 3: score every wig in the library against the face and rank them.

score(wig) = sum over shapes of  P(shape) x how well the wig suits that shape
"How well it suits" comes from common stylist guidance (e.g. round faces look
longer with length and a side fringe; long faces look shorter with a full fringe
and side volume). A wig's own `suitability` in the library overrides the rules.
"""
from .face import FaceMeasurements, decode_image, measure_face
from .library import Wig
from .shape import SHAPES, shape_probabilities

WEIGHTS = {"length": 0.35, "fringe": 0.30, "texture": 0.20, "volume": 0.15}

RULES = {
    "oval": {a: {} for a in WEIGHTS},  # suits almost anything -> default 0.8
    "round": {
        "length": {"pixie": .4, "short": .4, "bob": .3, "medium": .8, "long": .9},
        "fringe": {"none": .6, "side": .9, "full": .3},
        "texture": {"straight": .8, "wavy": .6, "curly": .4},
        "volume": {"low": .6, "medium": .8, "high": .5},
    },
    "square": {
        "length": {"pixie": .4, "short": .5, "bob": .4, "medium": .8, "long": .8},
        "fringe": {"none": .5, "side": .9, "full": .5},
        "texture": {"straight": .4, "wavy": .9, "curly": .8},
        "volume": {"low": .5, "medium": .8, "high": .7},
    },
    "oblong": {
        "length": {"pixie": .5, "short": .7, "bob": .9, "medium": .8, "long": .3},
        "fringe": {"none": .3, "side": .6, "full": .9},
        "texture": {"straight": .4, "wavy": .8, "curly": .9},
        "volume": {"low": .3, "medium": .7, "high": .9},
    },
    "heart": {
        "length": {"pixie": .6, "short": .6, "bob": .9, "medium": .8, "long": .6},
        "fringe": {"none": .5, "side": .9, "full": .6},
        "texture": {"straight": .6, "wavy": .8, "curly": .7},
        "volume": {"low": .6, "medium": .8, "high": .5},
    },
    "diamond": {
        "length": {"pixie": .5, "short": .6, "bob": .9, "medium": .8, "long": .6},
        "fringe": {"none": .4, "side": .8, "full": .9},
        "texture": {"straight": .6, "wavy": .8, "curly": .7},
        "volume": {"low": .5, "medium": .8, "high": .7},
    },
}
DEFAULT = 0.8


def attribute_scores(shape: str, wig: Wig) -> dict[str, float]:
    return {a: RULES[shape][a].get(getattr(wig, a), DEFAULT) for a in WEIGHTS}


def suitability(shape: str, wig: Wig) -> float:
    if shape in wig.suitability:
        return float(wig.suitability[shape])
    return sum(WEIGHTS[a] * s for a, s in attribute_scores(shape, wig).items())


def _reasons(shape: str, wig: Wig) -> list[str]:
    if shape in wig.suitability:
        return [f"picked by the catalogue team for {shape} faces"]
    if shape == "oval":
        return ["oval faces suit most styles; ranked on the closest other shapes"]
    good = [a for a, s in attribute_scores(shape, wig).items() if s >= 0.8]
    return [f"{getattr(wig, a)} {a} suits {shape} faces" for a in good] or [f"neutral match for {shape} faces"]


def rank_wigs(probs: dict[str, float], wigs: dict[str, Wig], top_k: int = 3,
              length: str | None = None, texture: str | None = None,
              colour: str | None = None, gender: str | None = None) -> list[dict]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    main_shape = max(probs, key=lambda s: probs[s])
    pool = [w for w in wigs.values()
            if (length is None or w.length == length)
            and (texture is None or w.texture == texture)
            and (colour is None or colour.lower() in (c.lower() for c in w.colours))
            and (gender is None or w.gender in (gender, "unisex"))]
    scored = [(sum(probs[s] * suitability(s, w) for s in SHAPES), w) for w in pool]
    scored.sort(key=lambda t: (-t[0], t[1].id))  # stable order on ties
    return [{"wig": w, "score": round(score, 4), "reasons": _reasons(main_shape, w)}
            for score, w in scored[:top_k]]


def recommend(image_bytes: bytes, wigs: dict[str, Wig], **kw) -> tuple[FaceMeasurements, dict, list[dict]]:
    face = measure_face(decode_image(image_bytes))
    probs = shape_probabilities(face)
    return face, probs, rank_wigs(probs, wigs, **kw)
