"""Cross-model disagreement — ensemble uncertainty without confidence scores.

Two different local models read the same bill independently. Where they
disagree — a line item one has and the other doesn't, a total that
differs, an amount that shifts — the disagreement IS the signal. No
confidence scores, no temperature tuning, no calibration data: the
architectural divergence between model families is the uncertainty
estimate.

Architecture:
  each model extracts independently (cached per model)
  the two extractions are aligned item-by-item
  every disagreement becomes a finding with both models' claims

The output is the strongest signal in the system: when gemma3:4b and
qwen2.5vl:3b both confidently state DIFFERENT totals for the same
printed bill, at least one is wrong and neither knows it.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .adjudicator import adjudicate
from .extract import extract_from_image
from .semantic import model_pulled, ollama_available
from .schemas import Extraction

AMOUNT_TOL = 0.02


def _norm(s: str) -> str:
    import re
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _cache_path(image_path: str, model: str) -> Path:
    safe = model.replace(":", "_").replace("/", "_")
    return Path(image_path).with_suffix("").with_suffix(
        f".{safe}.extraction.json")


def extract_cached(image_path: str, model: str) -> Extraction:
    """Extract with a per-model cache so comparisons are reproducible."""
    cp = _cache_path(image_path, model)
    if cp.exists():
        return Extraction.from_dict(json.loads(cp.read_text(encoding="utf-8")))
    ext = extract_from_image(image_path, model=model)
    cp.write_text(json.dumps(ext.to_dict(), indent=2), encoding="utf-8")
    return ext


@dataclass
class Disagreement:
    kind: str            # ITEM_ONLY_A | ITEM_ONLY_B | AMOUNT_DIFFERS | TOTAL_DIFFERS
    description_a: str | None = None
    description_b: str | None = None
    amount_a: float | None = None
    amount_b: float | None = None
    delta: float | None = None


@dataclass
class ComparisonReport:
    image: str
    model_a: str
    model_b: str
    total_a: float | None
    total_b: float | None
    disagreements: list = field(default_factory=list)
    agreed_items: int = 0
    verdict_a: dict = field(default_factory=dict)
    verdict_b: dict = field(default_factory=dict)

    @property
    def agree(self) -> bool:
        return not self.disagreements


def compare_models(image_path: str, model_a: str,
                   model_b: str) -> ComparisonReport:
    ext_a = extract_cached(image_path, model_a)
    ext_b = extract_cached(image_path, model_b)

    items_a = ext_a.line_items
    items_b = ext_b.line_items

    disagreements: list[Disagreement] = []
    used_b: set[int] = set()

    for ia in items_a:
        match = None
        for j, ib in enumerate(items_b):
            if j in used_b:
                continue
            if _norm(ia.description) == _norm(ib.description):
                match = j
                break
        if match is None:
            # try amount-only match (different wording, same number)
            for j, ib in enumerate(items_b):
                if j in used_b:
                    continue
                if abs(ia.amount_inr - ib.amount_inr) <= AMOUNT_TOL:
                    match = j
                    break
        if match is None:
            disagreements.append(Disagreement(
                kind="ITEM_ONLY_A", description_a=ia.description,
                amount_a=ia.amount_inr))
            continue
        ib = items_b[match]
        used_b.add(match)
        if abs(ia.amount_inr - ib.amount_inr) > AMOUNT_TOL:
            disagreements.append(Disagreement(
                kind="AMOUNT_DIFFERS",
                description_a=ia.description, description_b=ib.description,
                amount_a=ia.amount_inr, amount_b=ib.amount_inr,
                delta=round(ib.amount_inr - ia.amount_inr, 2)))

    for j, ib in enumerate(items_b):
        if j not in used_b:
            disagreements.append(Disagreement(
                kind="ITEM_ONLY_B", description_b=ib.description,
                amount_b=ib.amount_inr))

    agreed = sum(1 for d in disagreements if d.kind != "AMOUNT_DIFFERS")

    if (ext_a.stated_total_inr is not None
            and ext_b.stated_total_inr is not None
            and abs(ext_a.stated_total_inr - ext_b.stated_total_inr) > AMOUNT_TOL):
        disagreements.append(Disagreement(
            kind="TOTAL_DIFFERS",
            amount_a=ext_a.stated_total_inr, amount_b=ext_b.stated_total_inr,
            delta=round(ext_b.stated_total_inr - ext_a.stated_total_inr, 2)))

    va = adjudicate(ext_a)
    vb = adjudicate(ext_b)
    return ComparisonReport(
        image=os.path.basename(image_path), model_a=model_a, model_b=model_b,
        total_a=ext_a.stated_total_inr, total_b=ext_b.stated_total_inr,
        disagreements=disagreements,
        agreed_items=len(items_a) - sum(1 for d in disagreements
                                        if d.kind == "ITEM_ONLY_A"),
        verdict_a={"reconciled": va.reconciled, "summary": va.summary()},
        verdict_b={"reconciled": vb.reconciled, "summary": vb.summary()})


def render_comparison(rep: ComparisonReport) -> str:
    lines = []
    head = "AGREE" if rep.agree else "DISAGREE"
    color = "\033[92m" if rep.agree else "\033[91m"
    reset = "\033[0m"
    lines.append(f"\n\033[1m{'=' * 64}{reset}")
    lines.append(f"\033[1m{rep.image}{reset}  —  {color}{head}{reset}")
    lines.append(f"{'=' * 64}")
    lines.append(f"  {rep.model_a} stated: ₹{rep.total_a}")
    lines.append(f"  {rep.model_b} stated: ₹{rep.total_b}")
    if rep.disagreements:
        lines.append(f"\n  \033[1m{len(rep.disagreements)} disagreement(s):{reset}")
        for d in rep.disagreements:
            if d.kind == "TOTAL_DIFFERS":
                lines.append(f"    TOTAL_DIFFERS: {rep.model_a} says ₹{d.amount_a}, "
                             f"{rep.model_b} says ₹{d.amount_b} (Δ ₹{d.delta})")
            elif d.kind == "AMOUNT_DIFFERS":
                lines.append(f"    AMOUNT_DIFFERS '{d.description_a}': "
                             f"₹{d.amount_a} vs ₹{d.amount_b} (Δ ₹{d.delta})")
            elif d.kind == "ITEM_ONLY_A":
                lines.append(f"    ONLY {rep.model_a}: '{d.description_a}' "
                             f"₹{d.amount_a}")
            elif d.kind == "ITEM_ONLY_B":
                lines.append(f"    ONLY {rep.model_b}: '{d.description_b}' "
                             f"₹{d.amount_b}")
    else:
        lines.append("\n  both models extracted identical items and totals")
    lines.append(f"\n  adjudication under {rep.model_a}: "
                 f"{rep.verdict_a['summary']}")
    lines.append(f"  adjudication under {rep.model_b}: "
                 f"{rep.verdict_b['summary']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="elenchus-compare",
        description="Cross-model disagreement: two local models read the "
                    "same bill; their disagreement is the signal")
    ap.add_argument("paths", nargs="+", help="bill image(s)")
    ap.add_argument("--a", default="gemma3:4b")
    ap.add_argument("--b", default="qwen2.5vl:3b")
    args = ap.parse_args(argv)

    if not ollama_available():
        print("ollama not reachable")
        return 2
    for m in (args.a, args.b):
        if not model_pulled(m):
            print(f"model {m} not pulled — run: ollama pull {m}")
            return 2

    reports = []
    for p in args.paths:
        if not os.path.isfile(p):
            print(f"missing: {p}")
            continue
        rep = compare_models(p, args.a, args.b)
        print(render_comparison(rep))
        reports.append(asdict(rep))

    n_disagree = sum(1 for r in reports if r["disagreements"])
    print(f"\n{'=' * 64}")
    print(f"{len(reports)} bill(s) compared: {n_disagree} with disagreement, "
          f"{len(reports) - n_disagree} in full agreement")
    print("Every disagreement is a claim at least one model got wrong — "
          "and neither model flagged it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
