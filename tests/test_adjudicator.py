"""Elenchus test suite.

Run: python -m pytest tests/ -v

The headline test embeds the MEASURED gemma3:4b behavior: at temp=0,
three seeds, the model extracted 3 line items summing to 1845.68 while
stating a total of 1845.78 — a reproducible ₹0.10 mis-addition.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elenchus.adjudicator import adjudicate
from elenchus.schemas import Extraction, LineItem


def ext(items, total):
    return Extraction(
        line_items=[LineItem(d, a) for d, a in items],
        stated_total_inr=total)


class TestMeasuredFailure:
    def test_reproducible_paise_difference(self):
        """The measured gemma3:4b behavior, temp=0, seeds 42/7/2026.

        The paise difference (₹0.10) is WITHIN rounding tolerance and is
        correctly reported as INFO, not a false CRITICAL. The honest claim
        is: the model's stated total does not equal the sum of its own
        line items, deterministically, at temp=0.
        """
        e = ext([("Energy Charges", 1463.10),
                 ("Fixed Charge", 128.00),
                 ("Duty (Electricity Duty 16%)", 254.58)],
                1845.78)
        v = adjudicate(e)
        assert v.computed_total == 1845.68
        assert v.delta == 0.10
        codes = [f.code for f in v.findings]
        assert "TOTAL_ROUNDING" in codes
        assert "TOTAL_MISMATCH" not in codes
        assert v.critical_count == 0

    def test_naive_blind_sum_is_wrong_by_213(self):
        """The money shot: blind-trust arithmetic on the model's raw output.

        Measured: the naive prompt made the model scrape 6 metadata rows
        into line_items. Summing everything it emitted gives ₹2,109.68
        against a stated ₹1,895.78 — ₹213.90 wrong. Any pipeline that
        trusts the JSON without adjudication gets this number.
        """
        e = ext([("Consumer No.", 0), ("Name", 0), ("Address", 0),
                 ("Reading Date", 0), ("Units Consumed", 214),
                 ("Rate Slab", 0), ("Energy Charges", 1463.1),
                 ("Fixed Charge", 128.0), ("Duty (Electricity Duty 16%)", 254.58),
                 ("Due Date", 0), ("Penalty if late", 50)],
                1895.78)
        v = adjudicate(e)
        assert v.naive_total == 2109.68
        assert v.naive_delta == -213.90
        codes = [f.code for f in v.findings]
        assert "NAIVE_SUM_WRONG" in codes
        assert "METADATA_AS_CHARGES" in codes
        # the repair: 'Penalty if late' is conditional (you pay it only if
        # late), so it is filtered with the metadata; the charge-rows sum
        # is 1463.10 + 128.00 + 254.58 = 1845.68.
        assert v.computed_total == 1845.68
        assert v.delta == 50.10
        # and the headline: blind trust is off by ₹213.90; adjudication
        # narrows the error to the single ₹50.10 conditional-penalty
        # question instead of a ₹213.90 blind miss.
        assert abs(v.naive_delta) > 200
        assert abs(v.delta) < abs(v.naive_delta)


class TestCleanBill:
    def test_reconciled(self):
        e = ext([("Item A", 100.00), ("Item B", 50.50)], 150.50)
        v = adjudicate(e)
        assert v.reconciled is True
        assert v.findings == []

    def test_paise_rounding_is_info_not_critical(self):
        e = ext([("Item A", 100.00)], 100.40)
        v = adjudicate(e)
        assert v.reconciled is True
        codes = [f.code for f in v.findings]
        assert "TOTAL_ROUNDING" in codes
        assert "TOTAL_MISMATCH" not in codes
        assert v.critical_count == 0


class TestFailureModes:
    def test_confident_but_empty(self):
        """The 3/3 measured completeness hole: no items, confident total."""
        e = ext([], 1895.78)
        v = adjudicate(e)
        assert v.reconciled is False
        codes = [f.code for f in v.findings]
        assert "CONFIDENT_BUT_EMPTY" in codes
        assert v.critical_count >= 1

    def test_all_zero_amounts(self):
        e = ext([("Something", 0), ("Other", 0)], 500)
        v = adjudicate(e)
        codes = [f.code for f in v.findings]
        assert "ALL_ZERO_AMOUNTS" in codes

    def test_large_discrepancy_bank_statement(self):
        """The measured bank-statement class of error: ₹48k off."""
        e = ext([("Deposit", 100000.00), ("Deposit", 140600.00)], 192270)
        v = adjudicate(e)
        assert v.reconciled is False
        assert abs(v.delta) > 40000
        codes = [f.code for f in v.findings]
        assert "TOTAL_MISMATCH" in codes

    def test_negative_amounts_flagged(self):
        e = ext([("Purchase", 100.0), ("Refund", -50.0)], 50.0)
        v = adjudicate(e)
        codes = [f.code for f in v.findings]
        assert "NEGATIVE_AMOUNT" in codes

    def test_no_total_is_unverifiable(self):
        e = ext([("Item A", 100.0)], None)
        v = adjudicate(e)
        assert v.reconciled is False
        assert "UNVERIFIABLE" in v.summary()


class TestDeterminism:
    def test_same_input_same_output(self):
        e = ext([("A", 10.0), ("B", 20.0)], 35.0)
        v1 = adjudicate(e)
        v2 = adjudicate(e)
        assert v1.summary() == v2.summary()
        assert [f.code for f in v1.findings] == [f.code for f in v2.findings]
