# Elenchus

**The cross-examination. The model is the suspect.**

Constrained decoding guarantees **syntax**. It does not guarantee
**arithmetic**, **completeness**, or **calibration**. Elenchus is the
deterministic cross-examination of a local model's own extraction — and
the measurement harness that proves the checks work.

```
  LLM proposes (fast, sloppy)
        │
        ▼
  deterministic Python adjudicates (slow, exact)
        │
        ▼
  the DISAGREEMENT is the product
```

## The five layers

| Layer | Command | What it measures |
|---|---|---|
| **VERIFY** | `elenchus audit <bill>` | does the model's own arithmetic hold? |
| **GENERATE** | `elenchus evaluate` (auto) | synthetic bills with planted defects + ground-truth sidecars |
| **ADMIT** | `elenchus evaluate` | extraction accuracy + adjudication recall/FP against planted truth |
| **COMPARE** | `elenchus compare <bills>` | two models read the same bill; their disagreement is the signal |
| **CALIBRATE** | `elenchus calibrate <bill>` | under degradation: does the model abstain, or fabricate? |

## Measured results (this machine, gemma3:4b + qwen2.5vl:3b, temp=0)

**Adjudication: 100% recall on all 5 defect classes, 0% false positives, 18/18 bills.**

| defect class | adjudicator recall |
|---|---|
| MISADDED_TOTAL | 100% |
| WRONG_TAX (printed % vs computed %) | 100% |
| MISSING_ITEM (total includes dropped row) | 100% |
| METADATA_IN_ITEMS (units value as a charge) | 100% |
| EMPTY_WITH_TOTAL (confident total, no items) | 100% |
| clean bills falsely flagged | 0% |

**Cross-model disagreement concentrates exactly on fabrication.**
gemma3:4b vs qwen2.5vl:3b on 18 bills: 14 agree, and **all 4 disagreements
are fabrication cases**. The killer case: one bill where gemma stated
₹289.84 and qwen stated ₹2,289.84 — **₹2,000 apart** — and qwen's own
adjudicator said RECONCILED, because its items summed to its own wrong
total. Single-model adjudication passed an error only cross-model
disagreement exposes.

**The model never abstains. 0/14 abstentions, 29% fabrication rate under
degradation.** Blur, crop, rotate, contrast, JPEG — at blur-6 the model
confidently stated ₹638.00 against a clean read of ₹3,124.10. Not once did
it return null. Accuracy on clean bills and calibration under degradation
are different properties, and only the second is safety-relevant.

## The measured failure modes (all reproduced, all citable)

1. **Paise mis-addition, deterministic**: stated total ≠ sum of its own
   line items, at temp=0, across seeds.
2. **Blind-trust summation**: naive prompts make the model scrape
   metadata rows as charges; summing its raw JSON is ₹213.90 off.
3. **Schema-pressure fabrication**: on a zero-printed-item bill, the model
   *invented* a charge row — description "260" (the units consumed) with
   the total attached. Required fields drive fabrication.
4. **Internally-consistent wrong tax**: label claims 16%, amount computed
   at 10% — only percentage recomputation sees it.
5. **Inverted verdicts under fabrication**: the same bill, two models,
   both wrong, opposite adjudication outcomes.

## The adjudicator's checks

| Code | Severity | What it catches |
|---|---|---|
| `TOTAL_MISMATCH` | critical | stated total ≠ sum of charge rows |
| `NAIVE_SUM_WRONG` | critical/warning | blind-trust summation error |
| `FABRICATED_CHARGE_ROW` | critical | numeric descriptions = metadata values scraped as charges |
| `PERCENTAGE_MISMATCH` | critical | printed % ≠ recomputed % |
| `CONFIDENT_BUT_EMPTY` | critical | a stated total with zero evidence |
| `ALL_ZERO_AMOUNTS` | critical | the shape of a bill without its values |
| `METADATA_AS_CHARGES` | warning | metadata rows among the charges |
| `NEGATIVE_AMOUNT` | warning | negative rows (legitimate only as credits) |
| `TOTAL_ROUNDING` | info | paise-level difference — deliberately NOT critical; crying wolf on ten paise is bad adjudicator behaviour |

Design principle: **abstention over false alarm.** The percentage check
abstains on multi-% bills (hospital GST lines) rather than guessing the
base.

## Quick start

```bash
pip install -e ".[server]"
ollama pull gemma3:4b          # required
ollama pull qwen2.5vl:3b       # for the cross-model layer

# the web UI — one process serves the page and the API
python -m uvicorn elenchus.app:app

# the ladder
python -m elenchus demo                              # offline replay
python -m elenchus audit fixtures/bill.png           # one bill
python -m elenchus evaluate --per-cell 3             # the full harness
python -m elenchus compare corpus/*.png              # cross-model
python -m elenchus calibrate corpus/en-none-0.png    # degradation study
```

## Why local is the architecture, not a feature

1. **The input is the private data.** A bill carries your consumer
   number, your address, your usage. Verifying it via a cloud API is the
   leak.
2. **The adjudicator must own the ground truth.** The entire value is the
   reconciliation layer; local makes it auditable — stdlib anyone can read.
3. **Unit economics.** A kirana shop auditing 300 bills a month cannot
   pay per-page API fees. A 3.4GB model on one laptop is ₹0 marginal.

## Layout

```
elenchus/
  adjudicator.py   9 deterministic checks, zero model calls (the product)
  generator.py     synthetic bills, planted defects, ground-truth sidecars
  evaluate.py      the measurement harness
  compare.py       cross-model disagreement
  calibrate.py     abstention vs fabrication under degradation
  extract.py       the only model-touching file (cached)
  schemas.py       flat-field extraction schema
  semantic.py      model availability + optional narration
  report.py        terminal report
  app.py           FastAPI + single-page web UI
  cli.py           audit / demo / evaluate / compare / calibrate
fixtures/          the original measured bill + cached extractions
corpus/            generated evaluation corpus (gitignored)
tests/             14 tests: measured behaviors encoded as assertions
```

## Test suite

```bash
python -m pytest tests/ -v
# 14 passed — the measured behaviors are encoded as tests, including the
# reproducible ₹0.10 paise difference and the ₹213.90 blind-trust error.
```

## Citations

- arXiv:2609.23582 (ARID) — "output validity transfers but field semantics do not"
- arXiv:2607.20492 (PhantomFill) — required JSON fields drive fabrication
- arXiv:2604.25359 (Structured Output Benchmark) — near-perfect schema
  compliance alongside 23.7–83.0% value accuracy

## Safety note

All bills in `fixtures/` and `corpus/` are **synthetic**, generated by
`generator.py`. No real personal data is in this repository.
