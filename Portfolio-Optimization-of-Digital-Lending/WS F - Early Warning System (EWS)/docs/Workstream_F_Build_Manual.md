# Workstream F — Early Warning System (EWS)
### Build manual (hand-off copy)

**Project:** Digital Lending — Portfolio Optimization
**Prepared by:** Workstream E (Segmentation)
**For:** the Workstream F builder and the AI assistant running this manual

---

## What this is

This manual turns the **E→F Handoff Memo** into a working early-warning system: a dynamic, point-in-time score that flags *performing* loans heading toward 90+ DPD, with usable lead time, refreshed monthly. When you finish, it passes **Gate 4** and hands a watchlist + threshold table to Workstreams G (optimization), I (recommendation) and H (dashboard).

It is written to be run by an AI assistant working alongside a human reviewer, the same way the data and segmentation work was built: one step at a time, showing the result and getting a thumbs-up before moving on.

## What you need before you start

Get these first; everything below refers to them.

- **`WS_F_handoff_memo.md`** — F's mandate, the three lessons, the Gate-4 criteria. The single source of truth for *what* F must do.
- **`E_final_segment_assignments.csv`** — the frozen 6-segment scheme (`loan_id`/`customer_id` → segment). Flag rates get reported against this.
- **The behavioural panel `behaviour_monthly`** — F's primary raw material: `cashflow_consistency`, `balance_volatility`, `credit_utilisation`, `nach_bounce_count`, `spending_shock_flag`, `bureau_inquiry_velocity`, `app_engagement`.
- **`repayments`** — the DPD/default timeline (the outcome).  **`loans`** (origination + outcome) and **`customers`** (profile).
- **`_true_latent_risk_VALIDATION_ONLY.parquet`** — sanity-check only. **Never a feature** (A-019); it is creditworthiness-oriented (higher = safer).

Read the handoff memo end to end before writing any code.

## Two rules that matter most

1. **Point-in-time or nothing.** To predict default at month *t*, use only signals observable strictly **before** *t*. Features must *lead* the outcome, not move with it. This is the same trap that made E's "C4" behavioural-distress group unusable as a static segment — F exists to operationalise that pattern *cleanly*, so do not re-introduce the leakage E avoided.
2. **If this manual and the handoff memo ever disagree, the memo wins.**

## How to work through it

Go step by step. Finish a step, show what you produced, get approval, then start the next one. The order matters here more than usual: the **label** and the **point-in-time frame** are defined before any feature is built, because a single look-ahead leak invalidates everything downstream and is invisible until the out-of-time test exposes it.

---

## Pre-flight (before any code)

**1. Confirm you understand the mandate.** You should be able to state, in one sentence each:
- the unit of prediction (a *performing* loan-month, MOB ≥ 3, not yet 90+);
- the label (does this loan reach 90+ DPD within the forward window);
- why behaviour is allowed here when it was forbidden in segmentation (F predicts *forward*; E described *origination*).

**2. Set up the environment.**
- Python 3.11+ with `numpy`, `pandas`, `pyarrow`, `scikit-learn`. A monotonic-capable GBM (`lightgbm` or `xgboost`) for Step 6.
- One fixed random seed for every split and model: reuse the project convention; record it in the manifest.

**3. Know what you owe at the end.** Gate 4 expects:
- an **EWS score** per performing loan-month (and a current-month watchlist snapshot);
- **trigger rules** (the discrete-rule layer) and a **green/amber/red threshold table** with expected flag volume, precision and lead time per band;
- a **validation pack**: out-of-time AUC/KS, lift by decile, precision/recall at the chosen threshold, and — the metric that matters — **average lead time in days** before 90+;
- **flag rates by segment**, with the behavioural-distress cohort surfaced as the top watchlist.

**Suggested layout:**

```
wsF/
  config.py            # seed, paths, label window H, MOB floor, band cut-offs
  frame/
    spine.py           # Step 2 — performing loan-month observation frame
    labels.py          # Step 3 — forward-window 90+ label, censoring rules
    features.py        # Step 4 — point-in-time behavioural features
  model/
    split.py           # Step 5 — out-of-time + within-time splits
    scorecard.py       # Step 6 — logistic / scorecard baseline
    gbm.py             # Step 6 — monotonic GBM
  validate/
    metrics.py         # Step 7 — AUC/KS, lift, precision/recall, LEAD TIME
    leakage.py         # Step 7 — point-in-time + out-of-time audit
  policy/
    thresholds.py      # Step 8 — RAG bands tied to collections capacity
    segment_overlay.py # Step 9 — flag rates by segment, watchlist
  out/
    scores/  triggers/  thresholds/  validation/  gate4_results
```

---

## The build, step by step

Each step lists its goal, what to implement, the inputs it draws on, and a clear "done when" so you and the reviewer agree it is finished.

### Step 1 — Config and skeleton

**Goal:** one place that holds every choice the model depends on.

Put into `config.py`: the seed; the **forward label window H** (default 3 months ≈ 60–90 days — start here, sensitivity-test in Step 7); the **MOB floor** (3, the seasoned basis); the point-in-time lag rule (features use months ≤ *t*, label uses months *t+1 … t+H*); the out-of-time cut date for the split; and starting RAG band cut-offs (you will tune these in Step 8). Set up the seeded generator and the file paths.

**Done when:** every choice lives in one config, and the runner can import it.

### Step 2 — The observation spine (performing loan-months)

**Goal:** define the unit of prediction before touching features or labels.

For each loan, emit one row per calendar month from **MOB 3** onward, **while the loan is still performing** (DPD < 90 at the start of that month) and still on book (not yet Closed/Written-off). A loan that has already hit 90+ leaves the spine — F watches loans *on the way* to default, not after. Carry `loan_id`, `customer_id`, `period_date`, `months_on_book`, and join the segment from `E_final_segment_assignments.csv`.

**Watch the BNPL censoring trap.** A 2-month BNPL loan can never reach 90+ and a fixed early-window default rate makes risky pools look safe (the survivorship illusion the handoff flags). Short-tenure loans contribute few or no eligible rows — that is correct; do not pad them.

**Done when:** the spine is one row per performing loan-month, MOB ≥ 3; no row exists at or after the loan's first 90+ month; every row has a segment.

### Step 3 — The label (forward-window 90+)

**Goal:** a clean, leakage-free target.

For each spine row at month *t*, set `label = 1` if the loan transitions to **90+ DPD within months t+1 … t+H** (sticky default per §9.1 of the data spec), else `0`. Drop rows where the forward window is **censored** — i.e. the loan's panel ends (snapshot, prepay, closure) before *t+H* and no 90+ occurred, so the outcome is unknown. Do **not** code unknown as 0; that fabricates good outcomes and depresses measured risk.

**Done when:** every retained row has a defined 0/1 outcome with a fully observed forward window; the base rate is plausible (a few percent of performing months, much higher inside Contain/Subprime segments).

### Step 4 — Point-in-time features (the heart of F)

**Goal:** turn the 7 behavioural signals into leading indicators, every one lagged to month *t* or earlier.

Build from the **C4 drivers first** (the pattern E found but couldn't act on), then the rest of the panel — all as *trends, deltas, streaks and volatility*, not just levels:
- rolling 3-month change in `cashflow_consistency` (falling = stress);
- `balance_volatility` spikes vs the loan's own trailing baseline;
- `nach_bounce_count` streaks (consecutive bounce months);
- partial-payment runs and first-bounce flags from `repayments` (state at *t*, never future);
- `credit_utilisation` creep (rising trend);
- `bureau_inquiry_velocity` acceleration;
- `spending_shock_flag` onset and recency.

Also lay down the **discrete trigger layer** alongside the model features — e.g. *two consecutive partial payments + a bounce in the same month* — so coverage doesn't depend on the model alone (handoff §3).

**Two leakage guards, applied mechanically:**
- A feature at month *t* may read `behaviour_monthly` and `repayments` rows with `period_date ≤ t` only. Build a single windowing helper and route every feature through it.
- `_true_latent_risk` is **never** read into the feature frame (A-019). Keep it in a separate file used only by the Step 7 audit.

**Done when:** every feature is reproducibly point-in-time; a spot-check on a known defaulter shows the features deteriorating in the months *before* the 90+ month, not in it.

### Step 5 — Splits (out-of-time is the real test)

**Goal:** splits that prove the model generalises forward, the way it will be used.

Build two splits: a **within-time** split (random, stratified by label and segment) for tuning, and the decisive **out-of-time** split — train on older origination cohorts/months, test on newer ones, using the cut date in config. The out-of-time test is the proof the backtest reviewer insisted on; the within-time number alone will flatter you. Keep all months of a given loan on the **same side** of every split (no loan leaks across train/test).

**Done when:** both splits exist; no `loan_id` appears in both train and test of either split; test cohorts are strictly later than train for the out-of-time split.

### Step 6 — Models (governable first)

**Goal:** a score a risk committee will trust, plus a stronger challenger.

1. **Scorecard / logistic baseline** — the primary, because it is transparent and explainable per flag. Bin or WOE-transform the leading features; fit; read the coefficients for sign sanity (falling cashflow → higher risk, etc.).
2. **Monotonic GBM challenger** — `lightgbm`/`xgboost` with **monotonic constraints** in the economically correct direction on each feature, so the model can't learn a perverse non-monotone shape that a committee can't defend.

Calibrate scores to probabilities so the threshold table in Step 8 means something. Prefer the scorecard unless the GBM beats it materially on out-of-time lead time *and* lift — accuracy that can't be governed is not an upgrade.

**Done when:** both models produce calibrated scores on the spine; coefficient/constraint signs are all economically sensible; the chosen primary is named with a one-line justification.

### Step 7 — Validation (lead time is the metric that matters)

**Goal:** prove the flag fires early, generalises, and isn't leaking.

On the **out-of-time** test, report: AUC and KS; **lift in the top deciles**; precision/recall across the score range; and the headline — **average lead time in days** between the first flag and the 90+ event for true positives. A model with great AUC but near-zero lead time has failed F's actual job.

Then audit:
- **Leakage / point-in-time:** confirm no feature used future rows; confirm features lead rather than coincide. As a sanity check only, look at alignment against `_true_latent_risk` (expect a sensible inverse relationship — higher creditworthiness, lower flag rate); never let it touch training.
- **Stability:** performance holds across origination cohorts and across bootstrap resamples — not driven by one vintage.
- **Realistic false positives are expected.** About a third of stress episodes self-resolve by design (A-046); some amber flags will cure without defaulting. That is honest behaviour, not a bug — it sets the ceiling on achievable precision and must be reflected in the band design, not "fixed."

**Done when:** out-of-time AUC/KS, lift, precision/recall and a meaningful lead time are reported honestly; the leakage audit is clean; performance is stable across cohorts and resamples.

### Step 8 — Threshold / RAG band design

**Goal:** turn the score into actions collections can actually run.

Define **green / amber / red** bands on the calibrated score. Set the red cut where precision and lead time justify proactive intervention; set amber as a watch tier. Tie the cut-offs to two real constraints, not to a round number:
- **Collections capacity** — red+amber flag volume per month must be within what the team can work.
- **The cost asymmetry** — a false positive (chasing a loan that would have cured) vs a missed default. Make this trade-off explicit so the CRO sees what each threshold buys.

Produce the **threshold table**: per band, expected monthly flag volume, precision, recall, and average lead time. Write the headline figures (flag volume, precision, lead time) into the controlled workbook so the report, deck and dashboard all read **one number from one source**.

**Done when:** RAG bands are defined and justified against capacity and the FP-vs-missed cost; the threshold table is complete and its numbers live in the controlled workbook.

### Step 9 — Segment overlay and watchlist

**Goal:** connect F back to E and hand a usable list downstream.

- Report **flag rates by segment**, with priority on **Contain — Personal Non-prime (#3)** and **Subprime BNPL (#6)**, the two postures E routed explicitly to monitoring.
- Surface the **behavioural-distress cohort** (E's ~66%-default group, defined by bounces/shocks/utilisation) as the **top watchlist** — this is the C4 pattern, now a live, leakage-clean score rather than a coincident label.
- Emit the current-month **watchlist + score** per loan/customer with its band, ready for G (which prices/limits off it), I (the policy lever "route Contain to early-warning monitoring"), and H (the Risk dashboard's EWS funnel, precision and lead-time tiles).

**Done when:** segment flag rates are tabulated; the watchlist is emitted with score, band and segment; the behavioural-distress cohort is identified at the top.

### Step 10 — Gate 4 review and hand-off

**Goal:** the go / no-go.

Apply the **Gate-4 acceptance test** from the handoff memo — PASS only if all four hold:
- **Lead time:** flags fire *before* default, not coincident (measured lag, reported in days).
- **No leakage:** point-in-time validated; passes the out-of-time test.
- **Actionable:** precision/recall usable at the chosen threshold; flag volume operationally manageable.
- **Stable:** performance holds across cohorts and resamples.

On PASS: freeze the score definition and thresholds, record the seed/spec version/metrics in the manifest, and hand the watchlist, trigger rules, threshold table and validation pack to G, I and H.

---

## Watch out for (the common ways this breaks)

- **A feature that coincides with default instead of leading it.** The single most damaging failure — it inflates AUC and produces near-zero lead time. This is the C4 trap. Route every feature through the point-in-time window helper.
- **Censored windows coded as "good."** If a loan's panel ends inside the forward window with no observed 90+, the outcome is *unknown*, not 0. Coding it 0 manufactures safety and biases the model.
- **Loan rows leaking across the split.** Months of the same loan on both sides of train/test will look like brilliant generalisation and mean nothing. Split by loan, not by row.
- **Tuning on the within-time number.** It always flatters. The out-of-time result governs every decision.
- **Threshold set to a round number.** Bands must be tied to collections capacity and the FP-vs-missed cost, or the system is un-runnable in practice.
- **`_true_latent_risk` creeping into features.** It is validation-only (A-019). One accidental join and the model is a fraud detector for the generator, not an EWS.
- **Chasing precision past the self-cure ceiling.** ~35% of stress episodes resolve on their own (A-046); some amber flags *should* cure. Design for it; don't optimise it away.
- **Numbers diverging from the report.** Flag volume, precision and lead time live in the controlled workbook — one number, one source.

## Determinism checklist

- One fixed seed for all splits and model fits; recorded in the manifest.
- The build order is fixed (Steps 2 → 9); spine → label → features → split → model.
- Re-running reproduces the same scores, thresholds and validation numbers — diff two runs to confirm.

## What you hand over

- The **EWS score** definition + current-month **watchlist** (loan/customer → score, band, segment).
- The **trigger rules** and the **green/amber/red threshold table** (flag volume, precision, recall, lead time per band).
- The **validation pack** (out-of-time AUC/KS, lift, precision/recall, average lead time in days; leakage and stability audits).
- **Flag rates by segment** with the behavioural-distress cohort at the top.
- A one-line Gate 4 verdict: PASS, or PASS-with-waivers plus the waiver notes.

---

*Input to this manual: the E→F Handoff Memo and the frozen 6-segment scheme from Workstream E, plus the Workstream C data tables. Implement the memo exactly; the one discipline that overrides convenience everywhere is point-in-time integrity — raise anything unclear rather than improvising a feature that might peek forward.*
