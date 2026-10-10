"""Synthetic bill generator with planted defects and ground truth.

The keystone of the evaluation ladder. Generates realistic bill images
with KNOWN ground truth, so extraction accuracy can be measured instead
of asserted. Every defect is planted deliberately and recorded in the
sidecar — the adjudicator's precision and recall are measurable.

Defect classes:
  NONE            the bill's arithmetic is internally consistent
  MISADDED_TOTAL  stated total != sum of line items (the headline class)
  WRONG_TAX       duty/tax percentage is not what the line items imply
  MISSING_ITEM    one line item is omitted from the printed items
  METADATA_IN_ITEMS  metadata rows appear as charge rows
  EMPTY_WITH_TOTAL  confident stated total, zero line items

Ground truth sidecars live beside each image as <name>.truth.json.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- fonts

FONT_CANDIDATES = {
    "en": ["C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/arial.ttf",
           "C:/Windows/Fonts/segoeui.ttf"],
    "devanagari": ["C:/Windows/Fonts/Nirmala.ttc"],
}

_font_cache: dict = {}


def get_font(script: str, size: int) -> ImageFont.FreeTypeFont:
    key = (script, size)
    if key in _font_cache:
        return _font_cache[key]
    font = None
    for path in FONT_CANDIDATES.get(script, []):
        if os.path.isfile(path):
            try:
                font = ImageFont.truetype(path, size)
                break
            except Exception:
                continue
    if font is None:
        # fall back to any available candidate
        for paths in FONT_CANDIDATES.values():
            for path in paths:
                if os.path.isfile(path):
                    try:
                        font = ImageFont.truetype(path, size)
                        break
                    except Exception:
                        continue
            if font:
                break
    if font is None:
        font = ImageFont.load_default()
    _font_cache[key] = font
    return font


# ---------------------------------------------------------------- strings

# Per-language label sets. Devanagari rendering is approximate without
# raqm shaping — the generator marks generated-Devanagari bills so the
# evaluation can weight them separately.
LABELS = {
    "en": {
        "title": "ELECTRICITY BILL",
        "consumer": "Consumer No.",
        "name": "Name",
        "address": "Address",
        "reading_date": "Reading Date",
        "units": "Units Consumed",
        "slab": "Rate Slab",
        "energy": "Energy Charges",
        "fixed": "Fixed Charge",
        "duty": "Electricity Duty",
        "total": "TOTAL AMOUNT PAYABLE",
        "due": "Due Date",
    },
    "hi": {
        "title": "बिजली बिल",
        "consumer": "उपभोक्ता संख्या",
        "name": "नाम",
        "address": "पता",
        "reading_date": "रीडिंग तिथि",
        "units": "खपत इकाइयाँ",
        "slab": "स्लैब दर",
        "energy": "ऊर्जा शुल्क",
        "fixed": "स्थिर शुल्क",
        "duty": "बिजली कर",
        "total": "कुल देय राशि",
        "due": "देय तिथि",
    },
    "mr": {
        "title": "वीज बिल",
        "consumer": "ग्राहक क्रमांक",
        "name": "नाव",
        "address": "पत्ता",
        "reading_date": "वाचन तारीख",
        "units": "वापरलेल्या एकके",
        "slab": "स्लॅब दर",
        "energy": "ऊर्जा शुल्क",
        "fixed": "स्थिर शुल्क",
        "duty": "वीज कर",
        "total": "एकूण देय रक्कम",
        "due": "देय तारीख",
    },
}


@dataclass
class LineItem:
    description: str
    amount_inr: float
    is_charge: bool = True   # False => metadata row planted among charges


@dataclass
class BillSpec:
    seed: int = 0
    language: str = "en"        # en | hi | mr
    defect: str = "NONE"        # see module docstring
    units: int | None = None
    bill_no: str | None = None


@dataclass
class GroundTruth:
    line_items: list = field(default_factory=list)   # what is PRINTED
    stated_total_inr: float | None = None            # what is PRINTED
    true_total_inr: float = 0.0                      # what it SHOULD be
    defect: str = "NONE"
    language: str = "en"
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["line_items"] = [asdict(i) if isinstance(i, LineItem) else i
                           for i in self.line_items]
        return d


# ---------------------------------------------------------------- generation

DEFECT_CLASSES = [
    "NONE",
    "MISADDED_TOTAL",
    "WRONG_TAX",
    "MISSING_ITEM",
    "METADATA_IN_ITEMS",
    "EMPTY_WITH_TOTAL",
]


def _truth_path_for(image_path: str) -> Path:
    return Path(image_path).with_suffix("").with_suffix(".truth.json")


def generate_bill(image_path: str, spec: BillSpec) -> GroundTruth:
    """Render a bill image and write its ground-truth sidecar."""
    rng = random.Random(spec.seed)
    labels = LABELS.get(spec.language, LABELS["en"])
    script = "devanagari" if spec.language in ("hi", "mr") else "en"

    # --- the honest arithmetic -------------------------------------
    units = spec.units or rng.choice([180, 214, 260, 312, 145, 388, 96, 221])
    rate = rng.choice([6.80, 7.10, 6.50, 7.45])
    energy = round(units * rate, 2)
    fixed = rng.choice([110.0, 128.0, 95.0])
    duty_pct = 16.0
    duty_base = energy + fixed
    duty = round(duty_base * duty_pct / 100.0, 2)
    true_total = round(energy + fixed + duty, 2)

    items = [
        LineItem(labels["energy"], energy),
        LineItem(labels["fixed"], fixed),
        LineItem(f"{labels['duty']} ({duty_pct:g}%)", duty),
    ]

    notes: list = []
    stated = true_total

    # --- plant the defect ------------------------------------------
    if spec.defect == "MISADDED_TOTAL":
        stated = round(true_total + rng.choice([0.10, 50.10, 213.90, -18.75]), 2)
        notes.append(f"stated total altered by {stated - true_total:+.2f}")
    elif spec.defect == "WRONG_TAX":
        # The realistic, catchable version: the printed duty LABEL claims
        # one percentage but the AMOUNT is computed at another. The bill
        # is internally consistent (stated total follows the amount), so
        # only a percentage-recomputation check can see it. (A consistent
        # bill at the wrong statutory rate is NOT detectable by arithmetic
        # — that requires a domain prior, and we do not fake that.)
        wrong_pct = rng.choice([10.0, 20.0])
        wrong_duty = round(duty_base * wrong_pct / 100.0, 2)
        # keep the honest label, change the amount
        items[2] = LineItem(f"{labels['duty']} ({duty_pct:g}%)", wrong_duty)
        stated = round(energy + fixed + wrong_duty, 2)
        notes.append(f"duty labelled {duty_pct:g}% but computed at {wrong_pct:g}%")
    elif spec.defect == "MISSING_ITEM":
        # The real-world version: a line item row is dropped from the
        # printout but the stated total still reflects the full bill.
        # stated != sum(printed items) is the detectable signal — which
        # is the TOTAL_MISMATCH check firing for a structural reason.
        # (Recomputing the total to match would make it internally
        # consistent and undetectable — that would test nothing.)
        dropped = items.pop(rng.randrange(len(items)))
        stated = true_total  # the TRUE total, computed with the dropped item
        notes.append(f"line item '{dropped.description}' (₹{dropped.amount_inr}) "
                     f"omitted from print; stated total still includes it")
    elif spec.defect == "METADATA_IN_ITEMS":
        # The real failure mode: metadata rows with a NON-ZERO number in
        # the amount column. 'Units Consumed 214' rendered as a charge row
        # is exactly what the model scrapes (measured on the real bill:
        # 6 metadata rows in the charge list, blind sum off by ₹213.90).
        # Zero-amount rows are trivially skipped and prove nothing.
        items.insert(1, LineItem(labels["units"], float(units), is_charge=False))
        items.insert(2, LineItem(labels["slab"], 0.0, is_charge=False))
        items.insert(3, LineItem(labels["reading_date"], 0.0, is_charge=False))
        stated = true_total
        notes.append("3 metadata rows planted among the charge rows, one "
                     "carrying the units value as its amount")
    elif spec.defect == "EMPTY_WITH_TOTAL":
        items = []
        stated = true_total
        notes.append("zero line items printed with a confident stated total")

    truth = GroundTruth(line_items=items, stated_total_inr=stated,
                        true_total_inr=true_total, defect=spec.defect,
                        language=spec.language, notes=notes)
    _render(image_path, labels, script, spec, items, stated, units, rng)
    with open(_truth_path_for(image_path), "w", encoding="utf-8") as f:
        json.dump(truth.to_dict(), f, indent=2)
    return truth


def _render(image_path: str, labels: dict, script: str, spec: BillSpec,
            items: list, stated: float, units: int, rng) -> None:
    """Draw the bill. Clean, high-contrast, VLM-readable."""
    W, H = 1000, 760
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f_title = get_font(script, 34)
    f_head = get_font(script, 19)
    f_body = get_font(script, 18)
    f_small = get_font(script, 15)
    f_total = get_font(script, 24)

    ink = (20, 20, 20)
    muted = (90, 90, 90)
    line = (180, 180, 180)

    y = 30
    d.text((40, y), labels["title"], font=f_title, fill=ink)
    y += 60
    d.line([(40, y), (W - 40, y)], fill=line, width=2)
    y += 18

    meta = [
        (labels["consumer"], str(rng.randint(1000000000, 9999999999))),
        (labels["name"], rng.choice(["R. Kumar", "S. Priya", "A. Sharma",
                                     "M. Reddy", "K. Iyer"])),
        (labels["address"], rng.choice([
            "12 MG Road, Bengaluru 560001",
            "45 Anna Salai, Chennai 600002",
            "8 Banjara Hills, Hyderabad 500034"])),
        (labels["reading_date"], "2026-09-28"),
        (labels["units"], str(units)),
    ]
    for k, v in meta:
        d.text((40, y), k, font=f_body, fill=muted)
        d.text((330, y), str(v), font=f_body, fill=ink)
        y += 28
    y += 12

    # line-item table
    d.text((40, y), "DESCRIPTION", font=f_small, fill=muted)
    d.text((760, y), "AMOUNT (INR)", font=f_small, fill=muted)
    y += 22
    d.line([(40, y), (W - 40, y)], fill=line, width=1)
    y += 14
    for it in items:
        d.text((40, y), str(it.description), font=f_body, fill=ink)
        d.text((760, y), f"{it.amount_inr:,.2f}", font=f_body, fill=ink)
        y += 30
    y += 8
    d.line([(40, y), (W - 40, y)], fill=line, width=1)
    y += 16

    d.text((40, y), labels["total"], font=f_total, fill=ink)
    d.text((720, y), f"{stated:,.2f}" if stated is not None else "",
           font=f_total, fill=ink)
    y += 44
    d.text((40, y), f"{labels['due']}: 2026-10-20", font=f_body, fill=muted)

    img.save(image_path, format="PNG")


def generate_corpus(dest_dir: str, n_per_cell: int = 3,
                    languages=("en",), defects=None) -> list:
    """Generate a corpus across the defect x language matrix.

    Returns the list of ground-truth dicts.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    if defects is None:
        defects = ["NONE", "MISADDED_TOTAL", "WRONG_TAX", "MISSING_ITEM",
                   "METADATA_IN_ITEMS", "EMPTY_WITH_TOTAL"]
    truths = []
    seed = 1000
    for lang in languages:
        for defect in defects:
            for i in range(n_per_cell):
                seed += 1
                name = f"{lang}-{defect.lower()}-{i}.png"
                path = dest / name
                spec = BillSpec(seed=seed, language=lang, defect=defect)
                t = generate_bill(str(path), spec)
                t_dict = t.to_dict()
                t_dict["image"] = name
                truths.append(t_dict)
    return truths
