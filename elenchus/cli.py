"""Elenchus CLI.

  elenchus audit <bill.png>     extract (cached) + adjudicate
  elenchus audit --replay <x>   replay cached extraction only (offline)
  elenchus demo                 the full demo: measured fixtures
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__
from .adjudicator import adjudicate
from .extract import cache_path_for, extract_with_cache
from .report import render_verdict
from .schemas import Extraction


def _audit_one(path: str, replay: bool) -> int:
    if replay:
        cp = cache_path_for(path)
        if not os.path.isfile(cp):
            print(f"no cached extraction for {path} — run without --replay first",
                  file=sys.stderr)
            return 2
        with open(cp, encoding="utf-8") as f:
            ext = Extraction.from_dict(json.load(f))
    else:
        ext = extract_with_cache(path)
    v = adjudicate(ext)
    print(render_verdict(os.path.basename(path), ext, v))
    return 0 if v.reconciled else 1


def _demo() -> int:
    """The demo: replay the measured fixtures. Fully offline."""
    here = os.path.dirname(os.path.abspath(__file__))
    fixtures = os.path.join(here, "..", "fixtures")
    fixtures = os.path.abspath(fixtures)

    print("\033[1m" + "=" * 64 + "\033[0m")
    print("\033[1mELENCHUS — the cross-examination\033[0m")
    print("The model is the suspect. Deterministic code checks its arithmetic.")
    print("\033[1m" + "=" * 64 + "\033[0m")
    print("\033[2mAll outputs below are cached — the demo runs with wifi off.\033[0m")

    worst = 0
    if os.path.isdir(fixtures):
        for fn in sorted(os.listdir(fixtures)):
            if fn.endswith(".extraction.json"):
                label = fn.replace(".extraction.json", "")
                with open(os.path.join(fixtures, fn), encoding="utf-8") as f:
                    ext = Extraction.from_dict(json.load(f))
                v = adjudicate(ext)
                print(render_verdict(label, ext, v))
                if not v.reconciled:
                    worst = 1
    else:
        print(f"\nfixtures/ not found at {fixtures}")
        return 2

    print()
    print("\033[2mConstrained decoding buys syntax, never arithmetic.\033[0m")
    print("\033[2mA 3.4GB local model mis-added a bill by ₹0.10 at temperature 0.\033[0m")
    return worst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="elenchus",
        description="The cross-examination: deterministic audit of a local "
                    "model's own extraction arithmetic.")
    ap.add_argument("--version", action="version",
                    version=f"elenchus {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("audit", help="audit a bill image")
    a.add_argument("path")
    a.add_argument("--replay", action="store_true",
                   help="replay the cached extraction (no model call)")

    sub.add_parser("demo", help="run the full offline demo")

    cp = sub.add_parser("compare", help="cross-model disagreement: two local "
                                        "models read the same bill")
    cp.add_argument("paths", nargs="+", help="bill image(s)")
    cp.add_argument("--a", default="gemma3:4b")
    cp.add_argument("--b", default="qwen2.5vl:3b")

    ev = sub.add_parser("evaluate", help="measure extraction + adjudication "
                                         "against planted ground truth")
    ev.add_argument("--models", default="gemma3:4b",
                    help="comma-separated ollama model names")
    ev.add_argument("--per-cell", type=int, default=3)
    ev.add_argument("--languages", default="en")
    ev.add_argument("--corpus", default="corpus")
    ev.add_argument("--out", default="EVALUATION.md")

    cal = sub.add_parser("calibrate", help="measure abstention vs "
                                           "fabrication under degradation")
    cal.add_argument("paths", nargs="+")
    cal.add_argument("--model", default="gemma3:4b")

    args = ap.parse_args(argv)

    if args.cmd == "demo":
        return _demo()

    if args.cmd == "compare":
        from .compare import main as compare_main
        return compare_main(["--a", args.a, "--b", args.b] + args.paths)

    if args.cmd == "calibrate":
        from .calibrate import main as calibrate_main
        return calibrate_main(["--model", args.model] + args.paths)

    if args.cmd == "evaluate":
        from .evaluate import run_evaluation
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        langs = tuple(l.strip() for l in args.languages.split(",") if l.strip())
        run_evaluation(models, args.corpus, args.per_cell, langs, args.out)
        return 0
    return _audit_one(args.path, args.replay)


if __name__ == "__main__":
    sys.exit(main())
