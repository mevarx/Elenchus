"""The evaluation harness — the keystone of the next level.

Turns Elenchus from a demo into evidence. Generates a corpus with known
ground truth, runs every available local model through extraction,
measures against the truth, and reports:

  extraction accuracy   — do the models' line items and totals match
                          what was actually printed?
  adjudication recall   — of the bills with planted defects, how many
                          did the adjudicator flag?
  adjudication precision— of the CLEAN bills, how many were falsely
                          flagged?

Run:  python -m elenchus evaluate [--models gemma3:4b,qwen2.5vl:3b]
                                    [--per-cell 3] [--languages en]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .adjudicator import adjudicate
from .extract import extract_from_image
from .semantic import ollama_available, model_pulled
from .generator import (DEFECT_CLASSES, generate_bill, BillSpec)
from .schemas import EXTRACTION_SCHEMA, Extraction

# tolerance for "the model got the number right": paise-level
AMOUNT_TOL = 0.02


def _norm_desc(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[₹(),./]", " ", s)
    return s.strip()


def _match_items(pred: list, truth: list) -> tuple[int, int, int]:
    """Greedy match predicted line items against truth.

    Returns (true_positives, n_truth, n_pred).
    """
    truth_left = list(truth)
    pred_left = list(pred)
    tp = 0
    for p in pred_left:
        best = None
        best_score = -1.0
        for t in truth_left:
            d_score = 1.0 if _norm_desc(p["description"]) == _norm_desc(t["description"]) else 0.0
            pa, ta = p["amount_inr"], t["amount_inr"]
            a_score = 1.0 if abs(pa - ta) <= AMOUNT_TOL else 0.0
            if d_score and a_score:
                score = 2.0
            elif a_score and abs(pa - ta) <= 1.0:
                score = 1.5
            elif d_score:
                score = 1.0
            else:
                score = 0.0
            if score > best_score:
                best_score = score
                best = t
        if best is not None and best_score >= 1.0:
            tp += 1
            truth_left.remove(best)
    return tp, len(truth), len(pred)


@dataclass
class CellResult:
    model: str
    language: str
    defect: str
    image: str
    seconds: float
    extraction_ok: bool
    items_tp: int = 0
    items_truth: int = 0
    items_pred: int = 0
    total_correct: bool = False
    stated_extracted: float | None = None
    truth_stated: float | None = None
    adjudicator_flagged: bool = False
    adjudicator_any_finding: bool = False
    model_faithful: bool = False
    truth_is_defect: bool = False
    notes: list = field(default_factory=list)

    @property
    def item_precision(self) -> float:
        return self.items_tp / self.items_pred if self.items_pred else 0.0

    @property
    def item_recall(self) -> float:
        return self.items_tp / self.items_truth if self.items_truth else (1.0 if self.items_tp else 0.0)


def evaluate_cell(model: str, image_path: str, truth: dict,
                  skip_model: bool = False) -> CellResult:
    """Evaluate one image with one model. Ground truth drives scoring."""
    r = CellResult(
        model=model, language=truth.get("language", "?"),
        defect=truth.get("defect", "?"), image=os.path.basename(image_path),
        seconds=0.0, extraction_ok=False,
        truth_stated=truth.get("stated_total_inr"),
        truth_is_defect=(truth.get("defect", "NONE") != "NONE"),
    )

    if skip_model:
        # no-model path: adjudicate against the truth itself (used for
        # pure adjudicator evaluation without extraction)
        pred = {"line_items": truth.get("line_items", []),
                "stated_total_inr": truth.get("stated_total_inr")}
        r.extraction_ok = True
    else:
        t0 = time.time()
        try:
            ext = extract_from_image(image_path, model=model)
        except Exception as e:
            r.notes.append(f"extraction failed: {e}")
            return r
        r.seconds = time.time() - t0
        r.extraction_ok = True
        pred = ext.to_dict()

    r.stated_extracted = pred.get("stated_total_inr")
    if (pred.get("stated_total_inr") is not None
            and truth.get("stated_total_inr") is not None):
        r.total_correct = abs(pred["stated_total_inr"]
                              - truth["stated_total_inr"]) <= AMOUNT_TOL

    truth_items = [i for i in truth.get("line_items", [])]
    pred_items = pred.get("line_items", [])
    tp, n_truth, n_pred = _match_items(pred_items, truth_items)
    r.items_tp, r.items_truth, r.items_pred = tp, n_truth, n_pred

    v = adjudicate(Extraction.from_dict(pred))
    r.adjudicator_flagged = not v.reconciled
    r.adjudicator_any_finding = len(v.findings) > 0
    # Was the MODEL faithful? Compare the extraction to the printed truth
    # EXACTLY — every item's amount must match within paise tolerance and
    # the stated total must match. A misread amount (measured: 1080.25 ->
    # 1808.25, one digit) is an unfaithful extraction even though the
    # description matched, and an adjudicator flag on it is CORRECT
    # behavior. (The fuzzy item matcher above matches on description
    # alone, which would wrongly mark such a misread as faithful.)
    r.model_faithful = _extraction_faithful(pred, truth)
    r.notes.extend(f.code for f in v.findings if f.severity == "critical")
    return r


def _extraction_faithful(pred: dict, truth: dict) -> bool:
    """Did the model reproduce the printed bill exactly?"""
    truth_items = truth.get("line_items", [])
    pred_items = pred.get("line_items", []) or []
    if len(pred_items) != len(truth_items):
        return False
    truth_by_desc = {str(i["description"]).strip().lower(): i["amount_inr"]
                     for i in truth_items}
    for p in pred_items:
        key = str(p.get("description", "")).strip().lower()
        if key not in truth_by_desc:
            return False
        if abs(float(p.get("amount_inr", 0)) - truth_by_desc[key]) > AMOUNT_TOL:
            return False
    if (pred.get("stated_total_inr") is not None
            and truth.get("stated_total_inr") is not None):
        if abs(pred["stated_total_inr"] - truth["stated_total_inr"]) > AMOUNT_TOL:
            return False
    return True


def run_evaluation(models: list, corpus_dir: str, per_cell: int,
                   languages: tuple, out_path: str | None) -> dict:
    corpus = Path(corpus_dir)
    corpus.mkdir(parents=True, exist_ok=True)

    # 1. generate the corpus (idempotent: truth files drive reuse)
    print(f"[1/4] generating corpus: {per_cell}/cell x "
          f"{len(languages)} lang(s) x {len(DEFECT_CLASSES)} defects "
          f"in {corpus_dir}")
    seed = 5000
    truth_files = []
    for lang in languages:
        for defect in DEFECT_CLASSES:
            for i in range(per_cell):
                seed += 1
                name = f"{lang}-{defect.lower()}-{i}.png"
                path = corpus / name
                if not path.exists():
                    generate_bill(str(path),
                                  BillSpec(seed=seed, language=lang,
                                           defect=defect))
                tf = path.with_suffix("").with_suffix(".truth.json")
                truth_files.append(tf)

    # 2. evaluate each model
    results: list[CellResult] = []
    for model in models:
        if not ollama_available():
            print(f"  ! ollama not reachable — skipping model runs")
            break
        if not model_pulled(model):
            print(f"  ! model {model} not pulled — skipping")
            continue
        print(f"[2/4] evaluating {model} over {len(truth_files)} bills…")
        n_err = 0
        for tf in truth_files:
            truth = json.loads(tf.read_text(encoding="utf-8"))
            img = str(tf).replace(".truth.json", ".png")
            if not os.path.isfile(img):
                continue
            r = evaluate_cell(model, img, truth)
            results.append(r)
            if not r.extraction_ok:
                n_err += 1
                print(f"    ERR {r.image:<42} {r.notes[0] if r.notes else 'extraction failed'}")
                if n_err == 1 and not ollama_available():
                    print("    ! ollama became unreachable — aborting this model")
                    break
                continue
            mark = "OK "
            print(f"    {mark} {r.image:<42} "
                  f"items {r.items_tp}/{r.items_truth} "
                  f"total={'✓' if r.total_correct else '✗'} "
                  f"flagged={'✓' if r.adjudicator_flagged else '✗'} "
                  f"{r.seconds:.1f}s")
        if n_err == len([t for t in truth_files]):
            print(f"  ! every extraction failed for {model} — is ollama running?")

    # 3. aggregate
    print("[3/4] aggregating")
    summary = aggregate(results)

    # 4. report
    print("[4/4] report")
    report_md = render_report(summary, results)
    if out_path:
        Path(out_path).write_text(report_md, encoding="utf-8")
        print(f"  written: {out_path}")
    print()
    print(report_md)
    return {"summary": summary,
            "results": [asdict(r) for r in results]}


def aggregate(results: list) -> dict:
    by_model: dict = {}
    for r in results:
        m = by_model.setdefault(r.model, {
            "n": 0, "extraction_ok": 0, "seconds": [],
            "items_tp": 0, "items_truth": 0, "items_pred": 0,
            "total_correct": 0, "total_known": 0,
            "defect_bills": 0, "defect_flagged": 0, "defect_any": 0,
            "clean_bills": 0, "clean_flagged": 0, "caught_model_error": 0,
            "by_defect": {},
        })
        m["n"] += 1
        m["extraction_ok"] += int(r.extraction_ok)
        if r.extraction_ok:
            m["seconds"].append(r.seconds)
        m["items_tp"] += r.items_tp
        m["items_truth"] += r.items_truth
        m["items_pred"] += r.items_pred
        if r.truth_stated is not None:
            m["total_known"] += 1
            m["total_correct"] += int(r.total_correct)
        if r.truth_is_defect:
            m["defect_bills"] += 1
            m["defect_flagged"] += int(r.adjudicator_flagged)
            m["defect_any"] += int(r.adjudicator_any_finding)
            byd = m["by_defect"].setdefault(r.defect, {"n": 0, "flagged": 0})
            byd["n"] += 1
            byd["flagged"] += int(r.adjudicator_flagged)
        else:
            m["clean_bills"] += 1
            # an adjudicator flag on a CLEAN bill is a false positive ONLY
            # when the model extracted faithfully — if the model misread
            # the bill, the flag is correct behavior (a real disagreement)
            if r.model_faithful:
                m["clean_flagged"] += int(r.adjudicator_flagged)
            elif r.adjudicator_flagged:
                m["caught_model_error"] += 1

    out = {}
    for model, m in by_model.items():
        secs = m["seconds"]
        out[model] = {
            "bills": m["n"],
            "extraction_success": m["extraction_ok"] / m["n"] if m["n"] else 0,
            "median_seconds": (sorted(secs)[len(secs) // 2] if secs else None),
            "item_precision": m["items_tp"] / m["items_pred"] if m["items_pred"] else 0,
            "item_recall": m["items_tp"] / m["items_truth"] if m["items_truth"] else 0,
            "stated_total_accuracy": (m["total_correct"] / m["total_known"]
                                      if m["total_known"] else 0),
            # strict: flagged as CRITICAL (not reconciled)
            "adjudication_recall": (m["defect_flagged"] / m["defect_bills"]
                                    if m["defect_bills"] else None),
            # any-finding: any finding at all (incl. paise-level INFO)
            "adjudication_recall_any": (m["defect_any"] / m["defect_bills"]
                                        if m["defect_bills"] else None),
            "adjudication_fp_rate": (m["clean_flagged"] / m["clean_bills"]
                                     if m["clean_bills"] else None),
            # flags on clean bills where the MODEL misread — correct
            # behavior, counted separately, not as FP
            "caught_model_error": m["caught_model_error"],
            "by_defect": {k: (v["flagged"] / v["n"] if v["n"] else 0)
                          for k, v in m["by_defect"].items()},
        }
    return out


def render_report(summary: dict, results: list) -> str:
    lines = ["# Elenchus evaluation report", ""]
    lines.append(f"{len(results)} bill-model evaluations. "
                 "Ground truth is planted by the generator; every number "
                 "below is measured, not asserted.")
    lines.append("")
    lines.append("| model | bills | extraction ✓ | median s | item P | item R "
                 "| total acc | adj. recall | adj. recall (any) | adj. FP | caught model err |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for model, s in summary.items():
        med = f"{s['median_seconds']:.1f}" if s["median_seconds"] else "—"
        ar = f"{s['adjudication_recall']:.0%}" if s["adjudication_recall"] is not None else "—"
        ara = f"{s['adjudication_recall_any']:.0%}" if s["adjudication_recall_any"] is not None else "—"
        fp = f"{s['adjudication_fp_rate']:.0%}" if s["adjudication_fp_rate"] is not None else "—"
        lines.append(
            f"| {model} | {s['bills']} | {s['extraction_success']:.0%} "
            f"| {med} | {s['item_precision']:.0%} | {s['item_recall']:.0%} "
            f"| {s['stated_total_accuracy']:.0%} | {ar} | {ara} | {fp} "
            f"| {s['caught_model_error']} |")
    lines.append("")
    lines.append("## Adjudicator recall by defect class")
    lines.append("")
    defects = sorted({r.defect for r in results})
    header = "| model | " + " | ".join(defects) + " |"
    lines.append(header)
    lines.append("|---|" + "---|" * len(defects))
    for model, s in summary.items():
        cells = []
        for d in defects:
            v = s["by_defect"].get(d)
            cells.append(f"{v:.0%}" if v is not None else "—")
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Reading this")
    lines.append("")
    lines.append("- **extraction ✓** — the model returned parseable JSON at all")
    lines.append("- **item P / item R** — predicted line items matching the "
                 "printed truth (description + amount)")
    lines.append("- **total acc** — stated total matches the printed total")
    lines.append("- **adj. recall** — of bills with a planted defect, the "
                 "fraction flagged CRITICAL (not reconciled)")
    lines.append("- **adj. recall (any)** — the fraction with ANY finding, "
                 "including paise-level INFO. A ₹0.10 mis-addition is "
                 "reported as INFO on purpose — crying wolf on ten paise "
                 "is bad adjudicator behaviour — so the two columns differ "
                 "by exactly the paise-class defects.")
    lines.append("- **adj. FP** — of clean bills the MODEL READ FAITHFULLY, the "
                 "fraction falsely flagged. Flags on clean bills the model "
                 "MISREAD are not FPs — they are correct behavior, counted "
                 "separately as 'caught model err'.")
    lines.append("- **caught model err** — clean bills where the model itself "
                 "misread the printout and the adjudicator correctly flagged "
                 "the resulting disagreement")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="elenchus-evaluate",
                                 description="Measure extraction and "
                                             "adjudication against ground truth")
    ap.add_argument("--models", default="gemma3:4b",
                    help="comma-separated ollama model names")
    ap.add_argument("--per-cell", type=int, default=3)
    ap.add_argument("--languages", default="en",
                    help="comma-separated: en,hi,mr")
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--out", default="EVALUATION.md")
    args = ap.parse_args(argv)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    langs = tuple(l.strip() for l in args.languages.split(",") if l.strip())
    run_evaluation(models, args.corpus, args.per_cell, langs, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
