# Workstream F — Early Warning System (EWS)

A behaviour-led, point-in-time early-warning model that flags *performing* loans
heading toward 90+ DPD, with usable lead time. Built across 10 steps; **Gate 4 PASS**,
fully reproducible from seed `20260603`.

## Layout

```
config.py                  the single rulebook (all parameters)
frame/spine.py             Step 2 — performing loan-month observation spine
frame/labels.py            Step 3 — forward-window 90+ label (status-aware censoring)
frame/features.py          Step 4 — point-in-time behavioural features
model/split.py             Step 5 — within-time + purged/embargoed OOT splits
model/models.py            Step 6 — monotonic WOE scorecard (primary) + GBM challenger
validate/validation.py     Step 7 — OOT validation, lead time, leakage audit, Gate 4
policy/thresholds.py       Step 8 — RAG bands from capacity + cost asymmetry
policy/segment_overlay.py  Step 9 — segment flag rates + distress cohort + watchlist
freeze.py                  Step 10 — determinism check + freeze manifest
audit.py                   parameter scrutiny (run any time)
docs/                      build manual (roadmap) + closeout memo
out/                       all generated artifacts (from a clean back-test)
```

## Run (from scratch)

Place the three input folders next to the code, then run in order:

```
inputs/   FINAL/   validation_only/      # source data (not bundled)

python -m frame.spine
python -m frame.labels
python -m frame.features
python -m model.split
python -m model.models
python -m validate.validation
python -m policy.thresholds
python -m policy.segment_overlay
python -m freeze
python audit.py
```

Requires: pandas, pyarrow, numpy, scikit-learn, lightgbm.

## Back-test result (from-scratch integration run, empty `out/`)

| | |
|---|---|
| spine rows | 273,665 |
| labelled rows | 220,903 (base rate 2.16%) |
| model features | 13 behaviour-led, 0 NaN, 0 sign mismatches |
| splits | no loan-overlap; OOT seam leak-free |
| **OOT AUC (clean dpd=0)** | **0.883** (KS 0.681, lift 7.6×) |
| lead time (OOS) | mean 108d / median 91d (window-capped floor) |
| watchlist | 4,994 loans; 87% of operating-book defaults caught |
| distress cohort | 17.6× lift, 87% land RED/ESCALATION |
| **Gate 4** | **PASS** (all 6 checks) |
| determinism | re-fit max Δ = 0.00 (bit-identical) |

## Parameter audit (audit.py)

**52 in place · 1 placeholder-by-design · 3 MISSING · 0 anomalies.**

The 3 MISSING are the Risk Consultant's business inputs, required to finalise the
RAG band cuts (currently driven by clearly-logged placeholders F-A050–A053):
`COLLECTIONS_MONTHLY_CAPACITY`, `COST_FALSE_POSITIVE`, `COST_MISSED_DEFAULT`.

Note: `config.RAG_BANDS` (red 0.40 / amber 0.15) is **vestigial** — the live cuts are
capacity-derived in `thresholds.py` (≈0.036 / 0.003). Reconcile or remove to avoid
ambiguity about the authoritative band definition.

## Open items for reviewer

1. Ratify the **purged + embargoed temporal OOT split** (standard credit-risk method; not loan-disjoint).
2. Supply the **3 business numbers** to finalise thresholds; AMBER volume then needs a nudge (score-tie overshoot).
3. Report the **0.883** clean-population AUC, not the 0.957 full-book figure (inflated by already-delinquent loans).
4. Watchlist snapshot is **2026-03** (latest full-window); production refreshes on the live month.
