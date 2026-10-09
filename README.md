# Elenchus

**The cross-examination. The model is the suspect.**

Constrained decoding guarantees **syntax**. It does not guarantee
**arithmetic**. A 3.4GB local model mis-added a bill by ₹0.10 at
temperature zero — deterministically, across seeds — and on a naive
prompt it summed its own output to a number **₹213.90 away** from the
total it had just stated.

Elenchus is the check nobody ships: a deterministic adjudicator that
cross-examines a local model's extraction against its own evidence.

```
  naive sum    : ₹2,109.68   ← what you get by trusting the model's JSON
  adjudicated  : ₹1,845.68   ← charge rows only — the repair
  blind error  : ₹-213.90
```

## The measured finding

Run on this machine, gemma3:4b via Ollama, `temperature=0`, three seeds
(42, 7, 2026), same document:

| Run | Model's line items | Sum of items | Model's stated total | Δ |
|---|---|---|---|---|
| charges-only prompt, seed 42 | 3 rows | ₹1,845.68 | ₹1,845.78 | ₹+0.10 |
| charges-only prompt, seed 7 | 3 rows | ₹1,845.68 | ₹1,845.78 | ₹+0.10 |
| charges-only prompt, seed 2026 | 3 rows | ₹1,845.68 | ₹1,845.78 | ₹+0.10 |
| naive prompt (first run) | 11 rows — **6 of them metadata** | ₹2,109.68 | ₹1,895.78 | **₹−213.90** |

Three further measured failure modes on the same model:

- **Completeness hole**: one document returned `line_items: []` while
  still emitting a confident `stated_total_inr` — **3/3 runs**.
- **Metadata contamination**: the model scraped *Consumer No.*, *Reading
  Date*, *Units Consumed*, *Rate Slab* into the line items.
- **Enum failure**: classification failed **0/2** on Indic-language text
  until the class list was added to the prompt, then **5/5**.

Schema-valid output. Wrong numbers. Every time.

## The architecture

```
  LLM proposes (fast, sloppy)
        │
        ▼
  deterministic Python adjudicates (slow, exact)
        │
        ▼
  the DISAGREEMENT is the product
```

The model layer is one file (`extract.py`). Everything downstream —
every check, every number in the demo — is stdlib arithmetic and string
matching. The demo replays cached extractions, so it runs with wifi off
and produces identical output every time.

## The checks

| Code | Severity | What it catches |
|---|---|---|
| `CONFIDENT_BUT_EMPTY` | critical | a stated total with zero line items — a number with no evidence |
| `TOTAL_MISMATCH` | critical | stated total ≠ sum of charge rows |
| `NAIVE_SUM_WRONG` | critical/warning | blind-trust summation error — the number every consumer of the JSON gets |
| `ALL_ZERO_AMOUNTS` | critical | the shape of a bill without its values |
| `METADATA_AS_CHARGES` | warning | document metadata scraped into line items |
| `NEGATIVE_AMOUNT` | warning | negative rows (legitimate only as explicit credits) |
| `TOTAL_ROUNDING` | info | paise-level difference within tolerance |

## Why local is the architecture, not a feature

1. **The input is the private data.** A bill carries your consumer
   number, your address, your usage. Sending it to a cloud OCR API to
   "extract" it is the leak — the verification must live where the data
   lives.
2. **The adjudicator must own the ground truth.** The entire value is the
   reconciliation layer; local makes it auditable — ~150 lines of stdlib
   anyone can read.
3. **Unit economics.** A kirana shop auditing 300 bills a month cannot
   pay per-page API fees. A 3.4GB model on one laptop is ₹0 marginal.

## Usage

```bash
pip install -e .

# the full demo — fully offline, cached fixtures
python -m elenchus demo

# audit a live bill (caches the extraction beside the fixture)
python -m elenchus audit fixtures/bill.png

# replay only — no model call
python -m elenchus audit fixtures/bill.png --replay
```

## Test suite

```bash
python -m pytest tests/ -v
# 10 passed — the measured behaviors are encoded as tests, including the
# reproducible ₹0.10 paise difference and the ₹213.90 blind-trust error.
```

## Safety note

`fixtures/bill.png` is a **synthetic** bill generated for this demo. No
real personal data is in this repository.

## Citations

- arXiv:2609.23582 (ARID) — "output validity transfers but field
  semantics do not"
- arXiv:2607.20492 (PhantomFill) — required JSON fields drive
  fabrication; open-weight models never spend the escape token
- arXiv:2604.25359 (Structured Output Benchmark) — near-perfect schema
  compliance alongside 23.7–83.0% value accuracy
