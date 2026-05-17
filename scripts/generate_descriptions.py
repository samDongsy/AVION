"""Generate per-class LLM descriptions via Gemini 2.5 Flash and cache to JSON."""
from __future__ import annotations

import argparse
from pathlib import Path

from avion.dataset import list_image_folder
from avion.llm_prompting import generate_with_gemini, save_class_descriptions


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, help="ImageFolder root.")
    p.add_argument("--out", required=True, help="Output JSON path.")
    p.add_argument("--n", type=int, default=30, help="Descriptions per class (paper uses up to 50).")
    p.add_argument("--model", default="gemini-2.5-flash")
    args = p.parse_args()

    _, classnames = list_image_folder(args.data_root)
    mapping = generate_with_gemini(classnames, n_per_class=args.n, model=args.model)
    save_class_descriptions(mapping, args.out)
    print(f"wrote {len(mapping)} classes to {args.out}")


if __name__ == "__main__":
    main()
