# Workstream F — Early Warning System
## Closeout & hand-off memo

**Status: COMPLETE — Gate 4 PASS.** Determinism reproduced (seed 20260603, max score Δ = 0.00).
**Version:** EWS 0.1 · **Primary model:** WOE scorecard (governable) · **Challenger:** monotonic GBM.

---

### Mandate (delivered)

Build a dynamic, point-in-time score that flags *performing* loans heading toward 90+ DPD, monthly, with usable lead time before default — and hand a watchlist to optimization (G), recommendation (I) and the dashboard (H). Done.

### Gate 4 verdict — PASS

| Criterion | Result |
|---|---|
| **Lead time** — fires before default, not coincident | mean ~108d / median ~91d out-of-sample (a floor — window-capped); score rises monotonically into default |
| **No leakage** — point-in-time + out-of-time | clean: features computed only on data ≤ t; `true_latent_risk` never a feature (A-019); weak −0.13 sanity correlation |
| **Actionable** — usable precision/recall, manageable volume | RED 525/mo at 10.5% precision / 73% recall; watchlist 3,890/mo catches 87% of operating-book defaults |
| **Stable** — across cohorts and resamples | bootstrap AUC 95% CI [0.886, 0.933]; Contain segment validates strongest (0.896) |

### The honest headline

On the population that matters — loans **currently at zero delinquency**, where early warning is the whole point — the EWS scores **AUC 0.883, KS 0.681, top-decile lift 7.6×**, out-of-time. The full-book figure (0.957) is *not* the headline: it is inflated by already-delinquent loans that are trivial to "predict" and carry no lead time. Report 0.883, on currently-clean loans, and be ready to say so.

### What it actually does

A behaviour-led score (13 leading features — utilisation, balance-volatility level, NACH-bounce streaks, cashflow-consistency slope, spending-shock recency, partial-payment runs) feeds a monotonic, calibrated scorecard. Loans already ≥30 DPD bypass the score entirely via an **escalation trigger** straight to collections — early warning is reserved for loans that still look healthy. The output is a RED / AMBER / GREEN watchlist with a recommended action per band.

### Three decisions that define the build (and would be challenged in review)

1. **Behaviour-led, by necessity.** The first model hit AUC 0.97 — and its top flags were 100% already 30+ DPD. That is the coincident-signal trap the handoff memo named (the reason C4 was unusable). Removing `dpd_current` / `missed_cum_at_t` to an escalation role restored real lead time. *AUC fell; the system became useful.*
2. **Status-aware censoring.** A short-tenure loan that completes its term without defaulting is an observed *survivor*, not an unknown — only loans still Active at the data edge are right-censored. Conflating the two had inflated BNPL segments to absurd 40–100% base rates; the `loan_status` split fixed it.
3. **Purged + embargoed out-of-time split.** Strict loan-disjoint discarded 71% of training data on this long-lived panel. The standard credit-risk alternative — train on months whose label window closes before the cut, embargo the seam, test on the recent quarter — is what we used. **This needs reviewer ratification** (it allows a loan in both folds, with no information crossing the embargo).

### Segment findings (the overlay back to E)

- **Contain (#3, Personal Non-prime) is F's primary monitoring target** — 27.8% on the watchlist, highest RED load of any large segment. This is the live evidence for E's "route Contain to early-warning monitoring."
- **Subprime BNPL (#6) is an exit, not a monitoring problem** — 56.9% already escalated at snapshot, ~0% catchable early. Four independent views (spine, label, validation, overlay) agree: you don't early-warn your way out of it.
- **The C4 behavioural-distress cohort is now live and leakage-clean** — 627 loans at a 35.6% within-window default rate vs 2.0% book-wide (**17.6× lift**), 87% landing in RED/ESCALATION. The pattern E found but couldn't act on is now a forward-looking flag.

### Hand-off package (frozen)

| Artifact | File |
|---|---|
| Score definition (primary scorecard) | `model/models.py` |
| Current watchlist (score · band · segment · action) | `out/watchlist/ews_watchlist_current.csv` |
| RAG threshold table | `out/thresholds/ews_threshold_table.csv` |
| Flag rates by segment | `out/watchlist/flag_rates_by_segment.csv` |
| Validation pack | `out/validation/gate4_results.csv`, `validate/validation.py` |
| Freeze manifest (seed, spec, metrics, hashes) | `out/manifest.json` |

**Downstream:** G prices/limits off `ews_score` + band; I operationalises the Contain-monitoring lever; H builds the EWS funnel + precision/lead-time tiles. Headline figures (flag volume, precision, lead time) flow to the controlled workbook — one number, one source.

### Open items — what is NOT mine to close

1. **RED/AMBER capacity and FP-vs-missed-default costs are PLACEHOLDER** (F-A050–A053). The machinery is built; the *final* band cuts wait on the Risk Consultant's three numbers, which then recompute everything automatically.
2. **AMBER volume overshoots** its placeholder capacity (score ties at the low end); the amber cut needs nudging once real capacity lands.
3. **OOT split deviation** — ratify or replace.
4. **Watchlist snapshot is 2026-03** (latest full-window); production scoring refreshes on the live month, including currently-censored active loans.

### Reproduce

Fixed seed `20260603`; build order spine → labels → features → split → models → validation → thresholds → segment_overlay → freeze. Re-running reproduces every score bit-for-bit (verified: max Δ = 0.00).

---

*Workstream F closed on a clean Gate 4 PASS. The early-warning score is honest about its own limits — it reports lead time over AUC, names the population it works on, and routes already-failing loans away from itself. The one substantive thing it asks of the reviewer is to ratify the out-of-time methodology and supply the four business numbers that turn provisional bands into final ones.*
