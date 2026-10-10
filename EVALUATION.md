# Elenchus evaluation report

18 bill-model evaluations. Ground truth is planted by the generator; every number below is measured, not asserted.

| model | bills | extraction ✓ | median s | item P | item R | total acc | adj. recall | adj. recall (any) | adj. FP |
|---|---|---|---|---|---|---|---|---|---|
| gemma3:4b | 18 | 100% | 3.7 | 94% | 88% | 89% | 100% | 100% | 0% |

## Adjudicator recall by defect class

| model | EMPTY_WITH_TOTAL | METADATA_IN_ITEMS | MISADDED_TOTAL | MISSING_ITEM | NONE | WRONG_TAX |
|---|---|---|---|---|---|---|
| gemma3:4b | 100% | 100% | 100% | 100% | — | 100% |

## Reading this

- **extraction ✓** — the model returned parseable JSON at all
- **item P / item R** — predicted line items matching the printed truth (description + amount)
- **total acc** — stated total matches the printed total
- **adj. recall** — of bills with a planted defect, the fraction flagged CRITICAL (not reconciled)
- **adj. recall (any)** — the fraction with ANY finding, including paise-level INFO. A ₹0.10 mis-addition is reported as INFO on purpose — crying wolf on ten paise is bad adjudicator behaviour — so the two columns differ by exactly the paise-class defects.
- **adj. FP** — of clean bills, the fraction falsely flagged (lower is better; a high value means the adjudicator cries wolf)