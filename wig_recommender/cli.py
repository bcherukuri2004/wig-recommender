"""Command line: python -m wig_recommender.cli PHOTO [--top 3] [--debug out.jpg]"""
import argparse
import sys
from pathlib import Path

import cv2

from .face import FaceError
from .library import load_library
from .recommend import recommend


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Recommend wigs for a face photo")
    ap.add_argument("photo", type=Path)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--debug", type=Path, help="save the photo with the detected face boxed")
    args = ap.parse_args(argv)

    try:
        face, probs, ranked = recommend(args.photo.read_bytes(), load_library(), top_k=args.top)
    except (FaceError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print("Face shape:", ", ".join(f"{s} {p:.0%}" for s, p in sorted(probs.items(), key=lambda t: -t[1])))
    for w in face.warnings:
        print("warning:", w)
    for i, r in enumerate(ranked, 1):
        print(f"{i}. {r['wig'].name} ({r['score']:.2f}) - {'; '.join(r['reasons'])}")

    if args.debug:
        img = cv2.imread(str(args.photo))
        x, y, w, h = face.box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.imwrite(str(args.debug), img)
    return 0


if __name__ == "__main__":
    sys.exit(main())
