# Elenchus evaluation report

36 bill-model evaluations. Ground truth is planted by the generator; every number below is measured, not asserted.

| model | bills | extraction ✓ | median s | item P | item R | total acc | adj. recall | adj. recall (any) | adj. FP | caught model err |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma3:4b | 36 | 100% | 3.2 | 94% | 95% | 100% | 100% | 100% | 0% | 1 |

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
- **adj. FP** — of clean bills the MODEL READ FAITHFULLY, the fraction falsely flagged. Flags on clean bills the model MISREAD are not FPs — they are correct behavior, counted separately as 'caught model err'.
- **caught model err** — clean bills where the model itself misread the printout and the adjudicator correctly flagged the resulting disagreement