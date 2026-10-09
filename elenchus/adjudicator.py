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
    real_rows = [i for i in items if not _is_metadata(i.description)]
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
