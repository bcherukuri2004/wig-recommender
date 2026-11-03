"""The preset wig library: load it from JSON and check every entry is valid."""
import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "wigs.json"

ATTRIBUTES = {
    "length": ("pixie", "short", "bob", "medium", "long"),
    "fringe": ("none", "side", "full"),
    "texture": ("straight", "wavy", "curly"),
    "volume": ("low", "medium", "high"),
}
GENDERS = ("women", "men", "unisex")


@dataclass(frozen=True)
class Wig:
    id: str
    name: str
    length: str
    fringe: str
    texture: str
    volume: str
    gender: str = "unisex"
    colours: tuple[str, ...] = ()
    image: str | None = None
    # Optional manual scores per face shape (0-1); overrides the stylist rules.
    suitability: dict = field(default_factory=dict, hash=False, compare=False)


def load_library(path: Path | str = DEFAULT_PATH) -> dict[str, Wig]:
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list) or not raw:
        raise ValueError("wig library must be a non-empty JSON list")
    wigs: dict[str, Wig] = {}
    for i, item in enumerate(raw):
        try:
            wig = Wig(**{**item, "colours": tuple(item.get("colours", ()))})
        except TypeError as e:
            raise ValueError(f"wig #{i}: {e}") from None
        for attr, allowed in ATTRIBUTES.items():
            if getattr(wig, attr) not in allowed:
                raise ValueError(f"wig {wig.id!r}: {attr} must be one of {allowed}")
        for shape, score in wig.suitability.items():
            if not 0 <= score <= 1:
                raise ValueError(f"wig {wig.id!r}: suitability[{shape}] must be 0-1")
        if wig.gender not in GENDERS:
            raise ValueError(f"wig {wig.id!r}: gender must be one of {GENDERS}")
        if wig.id in wigs:
            raise ValueError(f"duplicate wig id {wig.id!r}")
        wigs[wig.id] = wig
    return wigs
