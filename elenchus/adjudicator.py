"""The adjudicator. The product.

Every check here is deterministic stdlib. The model's claims are
cross-examined against its own evidence, and the DISAGREEMENT is the
output. The demo's entire credibility rests on this file containing no
model calls.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .schemas import Extraction

# Tokens that mark a row as document metadata, not a billed charge.
# The model routinely scrapes these into line_items (measured: 6 of 11
# rows on the first fixture run).
METADATA_MARKERS = (
    "consumer", "name", "address", "reading date", "rate slab",
    "units consumed", "due date", "penalty", "mobile", "phone",
    "account", "customer id", "bill no", "invoice no", "date",
    "gstin", "email", "meter", "tariff",
)

# A line item whose DESCRIPTION is (almost entirely) numeric is a metadata
# VALUE scraped as a charge. Measured: on a bill with zero printed line
# items, gemma3:4b fabricated one row — description "260" (the units
# consumed) with the total attached. Real charge rows are text.
import re as _re


def _is_numeric_description(description: str) -> bool:
    d = description.strip()
    if not d:
        return False
    stripped = _re.sub(r"[\s,₹.%/-]", "", d)
    return bool(stripped) and stripped.replace(".", "", 1).isdigit()


def _parse_pct(description: str) -> float | None:
    """Extract a percentage from a line label like 'Electricity Duty (16%)'."""
    m = _re.search(r"\(?\s*(\d+(?:\.\d+)?)\s*%\s*\)?", description)
    return float(m.group(1)) if m else None

# Repeating-decimal tolerance for paise-level rounding: 1 rupee buys
# everything below it; beyond that a "rounding" excuse stops working.
ROUNDING_TOLERANCE_INR = 1.00


@dataclass
class Finding:
    code: str            # machine-readable, stable
    severity: str        # "critical" | "warning" | "info"
    detail: str
    numbers: dict = field(default_factory=dict)


@dataclass
class Verdict:
    reconciled: bool
    computed_total: float
    stated_total: float | None
    delta: float | None
    naive_total: float = 0.0
    naive_delta: float | None = None
    findings: list = field(default_factory=list)

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "critical")

    def summary(self) -> str:
        if self.reconciled:
            return "RECONCILED — the model's arithmetic holds"
        if self.stated_total is None:
            return "UNVERIFIABLE — the model stated no total"
        return (f"MISMATCH — model claims ₹{self.stated_total:,.2f}, "
                f"its own line items sum to ₹{self.computed_total:,.2f} "
                f"(Δ ₹{self.delta:+,.2f})")


def _is_metadata(description: str) -> bool:
    d = description.strip().lower()
    return any(m in d for m in METADATA_MARKERS)


def adjudicate(ext: Extraction) -> Verdict:
    """Cross-examine an extraction. Pure function of its input."""
    findings: list[Finding] = []
    items = ext.line_items
    amounts = [i.amount_inr for i in items]

    # --- check 1: completeness -------------------------------------
    if not items and ext.stated_total_inr is not None:
        findings.append(Finding(
            code="CONFIDENT_BUT_EMPTY",
            severity="critical",
            detail=("the model stated a total but extracted zero line "
                    "items — a confident number with no evidence behind "
                    "it. Measured on gemma3:4b in 3/3 runs at temp=0."),
            numbers={"stated_total_inr": ext.stated_total_inr}))

    # --- check 2: metadata contamination ----------------------------
    metadata_rows = [i for i in items if _is_metadata(i.description)]
    # a fabricated charge: a purely numeric description (a metadata VALUE
    # scraped as a charge row). Measured: gemma3:4b fabricated
    # {description: "260", amount: <total>} on a zero-item bill.
    fabricated_rows = [i for i in items if _is_numeric_description(i.description)]
    real_rows = [i for i in items
                 if not _is_metadata(i.description)
                 and not _is_numeric_description(i.description)]
    if metadata_rows:
        findings.append(Finding(
            code="METADATA_AS_CHARGES",
            severity="warning",
            detail=(f"{len(metadata_rows)} row(s) look like document "
                    f"metadata, not billed charges, but the model put them "
                    f"in the line items: "
                    + "; ".join(f"'{i.description}' ₹{i.amount_inr:g}"
                                for i in metadata_rows[:4])),
            numbers={"metadata_rows": len(metadata_rows),
                     "total_rows": len(items)}))

    if fabricated_rows:
        findings.append(Finding(
            code="FABRICATED_CHARGE_ROW",
            severity="critical",
            detail=(f"{len(fabricated_rows)} 'charge' row(s) have a purely "
                    f"numeric description — a metadata VALUE scraped into "
                    f"the charge list, not a real billed item: "
                    + "; ".join(f"'{i.description}' = ₹{i.amount_inr:g}"
                                for i in fabricated_rows[:3])
                    + ". Real charge rows are text; this is schema-pressure "
                      "fabrication."),
            numbers={"fabricated_rows": len(fabricated_rows)}))

    # --- check 3: arithmetic ----------------------------------------
    # Two sums, deliberately:
    #   naive_total     — everything the model emitted, added up blindly.
    #                     This is what ANY consumer of the extraction gets
    #                     if they just trust the JSON. When metadata rows
    #                     contaminate the items, this number is wrong.
    #   computed_total  — only plausible charge rows. The repair.
    naive_total = round(sum(amounts), 2)
    basis = real_rows if metadata_rows else items
    computed = round(sum(i.amount_inr for i in basis), 2)

    if ext.stated_total_inr is not None and basis:
        delta = round(ext.stated_total_inr - computed, 2)
        if abs(delta) > ROUNDING_TOLERANCE_INR:
            findings.append(Finding(
                code="TOTAL_MISMATCH",
                severity="critical",
                detail=(f"the model's stated total (₹{ext.stated_total_inr:,.2f}) "
                        f"disagrees with the sum of its own line items "
                        f"(₹{computed:,.2f}) by ₹{delta:+,.2f}"),
                numbers={"stated_total_inr": ext.stated_total_inr,
                         "computed_total": computed,
                         "delta": delta}))
        elif abs(delta) > 0:
            findings.append(Finding(
                code="TOTAL_ROUNDING",
                severity="info",
                detail=(f"paise-level difference (₹{delta:+,.2f}) between the "
                        f"stated total and the sum of charge rows — flagged "
                        f"for completeness; within rounding tolerance"),
                numbers={"delta": delta}))

    # the blind-trust number: what you get by summing the model's raw output
    naive_delta = (round(ext.stated_total_inr - naive_total, 2)
                   if ext.stated_total_inr is not None else None)
    if (naive_delta is not None
            and abs(naive_delta) > ROUNDING_TOLERANCE_INR
            and abs(naive_delta - (delta if ext.stated_total_inr is not None
                                   and basis else 0)) > 0.005):
        findings.append(Finding(
            code="NAIVE_SUM_WRONG",
            severity="critical" if abs(naive_delta) > 10 else "warning",
            detail=(f"summing the model's raw line items blindly gives "
                    f"₹{naive_total:,.2f} — ₹{naive_delta:+,.2f} off the "
                    f"stated total. This is what every consumer of the "
                    f"extraction gets without adjudication."),
            numbers={"naive_total": naive_total,
                     "naive_delta": naive_delta}))

    # --- check 4: negative or absurd amounts -------------------------
    negatives = [i for i in items if i.amount_inr < 0]
    if negatives:
        findings.append(Finding(
            code="NEGATIVE_AMOUNT",
            severity="warning",
            detail=(f"{len(negatives)} negative line item(s) in a bill — "
                    "legitimate only as explicit credits/discounts"),
            numbers={"negatives": len(negatives)}))

    zero_only = items and all(a == 0 for a in amounts)
    if zero_only:
        findings.append(Finding(
            code="ALL_ZERO_AMOUNTS",
            severity="critical",
            detail=("every extracted line item has amount 0 — the model "
                    "produced the SHAPE of a bill without its values"),
            numbers={"rows": len(items)}))

    # --- check 5: printed percentage vs. arithmetic -------------------
    # If a line item carries a percentage label (e.g. 'Electricity Duty
    # (16%)'), recompute it against the other charge rows and flag the
    # disagreement. This is the domain-rule layer: it catches WRONG_TAX,
    # which pure sum-consistency cannot see (a bill can be internally
    # consistent AND have the wrong tax rate).
    if ext.stated_total_inr is not None and real_rows:
        pct_rows = [(i, _parse_pct(i.description)) for i in real_rows]
        pct_rows = [(i, p) for i, p in pct_rows if p is not None]
        if len(pct_rows) == 1:
            pct_item, pct = pct_rows[0]
            others = [i for i in real_rows if i is not pct_item]
            other_sum = round(sum(i.amount_inr for i in others), 2)
            if other_sum > 0:
                expected = round(other_sum * pct / 100.0, 2)
                if abs(expected - pct_item.amount_inr) > max(1.0, other_sum * 0.01):
                    findings.append(Finding(
                        code="PERCENTAGE_MISMATCH",
                        severity="critical",
                        detail=(f"'{pct_item.description}' claims {pct:g}% but "
                                f"₹{pct_item.amount_inr:,.2f} is not "
                                f"{pct:g}% of the other charges "
                                f"(₹{other_sum:,.2f} → ₹{expected:,.2f}). "
                                f"The bill is internally consistent but the "
                                f"printed rate does not match its own "
                                f"arithmetic."),
                        numbers={"printed_pct": pct,
                                 "printed_amount": pct_item.amount_inr,
                                 "expected_amount": expected,
                                 "base": other_sum}))

    reconciled = (ext.stated_total_inr is not None
                  and not any(f.severity == "critical" for f in findings))
    return Verdict(reconciled=reconciled,
                   computed_total=computed,
                   stated_total=ext.stated_total_inr,
                   delta=(round(ext.stated_total_inr - computed, 2)
                          if ext.stated_total_inr is not None else None),
                   naive_total=naive_total,
                   naive_delta=naive_delta,
                   findings=findings)
