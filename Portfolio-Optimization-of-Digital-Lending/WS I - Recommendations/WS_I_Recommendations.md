# Recommendations — Digital Lending Portfolio Optimization

**Prepared by:** Anwesha & Shreyan
**Date:** 12 June 2026
**Status:** Draft v1
**Inputs:** WS D (channel economics), WS E (segmentation), WS G (policy scenarios). WS F (EWS thresholds) pending — qualitative placeholder included.

---

## 1. Executive Summary

We recommend a four-lever portfolio reshape that delivers **+₹17.85 cr improvement in book contribution at baseline**, rising to **+₹19.01 cr under +3pp macro stress**. Three defensive levers shrink loss-making slices of the book; one offensive lever replaces most of the lost volume with higher-quality origination.

Net effect:

- Book size reshaped by approximately 4.2% (−₹58.83 cr origination)
- Portfolio contribution improved by +₹17.85 cr at baseline (+₹19.01 cr at +3pp stress)
- Recommendation *strengthens* under stress — the defensive levers dominate, so worse macro conditions amplify (not erode) the gain
- All four levers are operationally implementable using data already in the application pipeline

---

## 2. Strategic Context

Analysis of 49,600 loans (₹1,398 cr origination across six segments) reveals concentrated value destruction in three segments:

- **BNPL** (17,236 loans, ₹25.9 cr origination): 77.7% of loans loss-making. The damage concentrates in Digital ads and DSA channels (mean per-loan loss of −₹1,326 and −₹1,096; default rates 11.06% and 10.80%).
- **Personal Subprime D-E** (5,844 loans, ₹72.5 cr): segment profitable overall (+₹2.99 cr), but 12.6% of loans are loss-makers. A clean behavioural cutoff (`cashflow_consistency_mean < 0.6`) separates losers from winners.
- **SME Non-prime C-D-E** (5,461 loans, ₹536.4 cr): segment highly profitable overall (+₹22.37 cr), but 7% of loans loss-making with disproportionate rupee impact due to large ticket sizes (~₹9 lakh average). The same `cashflow_consistency` signal applies.

Meanwhile, **SME Prime A-B** (4,558 loans, +₹14.68 cr total contribution) is the cleanest segment in the book — only 2.6% loss-making, 2.74% default rate — and undersized relative to its unit economics.

The strategic posture this analysis supports: **shrink the worst, surgically clean the borderline, grow the best.**

---

## 3. Headline Recommendation — Contain SME Non-prime C-D-E

**Lever:** Decline new SME applications where `origination_risk_grade ∈ {C, D, E}` AND `cashflow_consistency_mean < 0.65`.

**Quantified impact:**

| Macro condition | Net Δ in book contribution |
|---|---|
| Baseline (0pp) | **+₹12.97 cr** |
| +2pp stress | +₹14.30 cr |
| +3pp stress | +₹14.96 cr |

**Operational scope:** ~950 applications declined annually, ₹88.32 cr origination foregone (17% of current SME C-D-E flow).

**Implementation owner:** SME Underwriting.

**Guardrail metric:** Monitor SME C-D-E decline rate weekly. Expected stable rate ~17-18% of incoming flow. Deviation >2pp from this band triggers re-calibration.

**Why it's the headline:**

1. Largest absolute impact (~7x Lever 1, ~5x Lever 2)
2. Monotonically increases under stress — the lever does its best work exactly when the macro environment turns
3. Surgical, not segment-wide — only 9% of SME C-D-E declined; the remaining 91% (~4,963 loans, profitable) continue unchanged
4. Uses signals already observable at the application window (grade from underwriting model, cashflow consistency from bank statements via account aggregator)

---

## 4. Supporting Lever 1 — Tighten Personal Subprime D-E

**Lever:** Decline new Personal applications where `origination_risk_grade ∈ {D, E}` AND `cashflow_consistency_mean < 0.6`.

**Impact:**
- Baseline: +₹2.33 cr
- +2pp stress: +₹2.44 cr
- +3pp stress: +₹2.50 cr

**Scope:** 767 loans declined annually, ₹9.39 cr origination foregone (~13% of Personal D-E flow).

**Owner:** Personal Underwriting.

**Guardrail:** Personal D-E decline rate stable at ~13% of incoming flow.

**Rationale:** Same behavioural signal as the headline; demonstrates the rule generalizes across product lines, not just SME.

---

## 5. Supporting Lever 2 — Cease BNPL via Digital Ads and DSA

**Lever:** Discontinue BNPL acquisition through Digital ads and DSA channels. Existing book runs off naturally; no early write-off.

**Impact:**
- Baseline: +₹1.08 cr
- +2pp stress: +₹1.24 cr
- +3pp stress: +₹1.31 cr

**Scope:** 8,610 loans/year no longer acquired, ₹12.95 cr origination foregone.

**Owner:** BNPL Marketing + DSA Channel Management.

**Guardrail:** Monitor remaining BNPL channels (Partner-embedded, Referral, Organic) for volume drift and unit-economic deterioration. If mean per-loan value_proxy in any of these turns more negative than −₹600, escalate.

**Rationale:** Digital ads (−₹1,326/loan) and DSA (−₹1,096/loan) are structurally loss-making. CAC paid through these channels exceeds the loans' lifetime contribution. Pause acquisition is cleaner than re-pricing because the unit economics are too far underwater to fix with APR alone.

---

## 6. Offensive Lever — Grow SME Prime A-B by 10%

**Lever:** Increase SME A-B acquisition by ~456 loans annually (10% of current 4,558).

**Impact:**

| Macro condition | Net Δ |
|---|---|
| Baseline (0pp) | +₹1.47 cr |
| +2pp stress | +₹0.69 cr |
| +3pp stress | +₹0.30 cr |

**Scope:** 456 new loans, ₹51.83 cr origination added.

**Owner:** SME Sales + Business Development.

**Guardrail:** SME A-B default rate must stay below 3.0% as volume grows. Re-evaluate growth target quarterly.

**Note on asymmetry:** Unlike the three defensive levers, this lever weakens under stress because new loans take stressed losses. We include it because:

1. It replaces ~47% of volume given up by the defensive levers (₹51.83 cr added vs ~₹110.66 cr removed), making the recommendation a *reshape* rather than a *shrink*.
2. SME A-B is the only segment where growth is safe under stress — its 2.74% default rate and ₹32,204 per-loan margin provide enough cushion that the lever stays positive even at +3pp.
3. Without this lever, every other segment's growth would actively destroy value under stress. Demonstrating that we modelled this is itself a signal to the CRO of analytical rigour.

---

## 7. Supporting Lever — Early Warning System

**Status:** Operational. Workstream F complete, Gate 4 PASS (all 6 checks). Deployed as the in-life detection layer across the entire performing book; deterministic, seed-locked, validated out-of-time on a purged + embargoed temporal split.

**Description:** A behavioural Early Warning System (EWS) flags loans drifting toward default before they hit 90+ DPD. Built as a monotonic WOE scorecard (primary, governable) with a GBM challenger, scoring ~17,000 performing loans monthly across 13 behaviour-led features grouped into five clusters:

- **Cashflow signals** — `cashflow_consistency_level`, `cashflow_consistency_slope`
- **NACH bounce signals** — `bounce_streak`, `bounce_count_3m`, `first_bounce_flag`, `partial_run`
- **Utilisation / balance signals** — `utilisation_level`, `utilisation_creep_3m`, `balance_volatility_level`
- **Spending shock signals** — `spending_shock_recency`, `spending_shock_rate_3m`
- **Engagement signals** — `inquiry_velocity_accel`, `app_engagement_drop`

**Operational bands (per WS F threshold calibration):**

| Band | Volume / month | % of book | Recall | Action |
|---|---|---|---|---|
| ESCALATION (dpd ≥ 30) | 977 | 5.7% | bypass | Direct to collections; no scoring required |
| RED (score ≥ 0.036) | 525 | 3.1% | 73% | Agent call within 7 days; restructure offer where appropriate |
| AMBER (0.003 – 0.036) | 3,365 | 19.7% | 13.7% | Soft SMS / email outreach; weekly monitoring |
| GREEN (< 0.003) | 13,158 | 77.2% | — | No action |

The combined RED + AMBER watchlist of **3,890 loans/month** captures **87% of operating-book defaults**.

**Performance (Gate 4 PASS):**

- AUC **0.883** on currently-clean loans (dpd = 0); bootstrap 95% CI [0.886, 0.933]
- KS **0.681**; top-decile lift **7.6×**
- Lead time: **median 91 days, mean 108 days** — window-capped, so this is the floor, not the ceiling
- Validates strongest where Contain runs hardest: **Personal Non-prime AUC 0.896**
- Distress cohort (627 loans at 35.6% within-window default rate): **17.6× lift**, 87% land in RED or Escalation

We deliberately reject the full-book AUC of 0.957 as the headline metric: it is inflated by already-delinquent loans whose label is mechanically determinable from `dpd_current`. The 0.883 figure on currently-clean loans is the honest reflection of in-life detection power.

**Why included alongside the four underwriting levers:** The four levers above act *at underwriting* — they prevent bad loans from being made. The EWS acts *during the loan lifecycle* — it catches loans that pass underwriting but deteriorate later. The two failure modes are different; both need addressing. The EWS additionally serves as the operational backstop when intake cutoffs let a borderline case through.

**Quantified impact:** WS F validated detection performance comprehensively but did not compute a rupee loss-avoidance figure. Translating recall × lead-time into avoided loss severity requires a separate cost-of-delay calculation that has not been run. The provisional belief from earlier scoping — ~₹3–5 cr of additional avoided losses annually (~30–50% reduction in loss severity for flagged loans) — therefore remains qualitative pending that calculation. The defensible operational numbers today are: 87% recall on operating-book defaults, 91-day median lead time, 3,890-loan watchlist, and 73% recall in the high-confidence RED band.

**Shared infrastructure with Levers 2 and 5:** The EWS uses `cashflow_consistency_level` as one of its 13 features — the same signal that powers the L2 and L5 intake cutoffs. The signal pipeline (account-aggregator → cashflow_consistency derivation) is therefore shared across three levers. An outage of this pipeline degrades the EWS partially (12 other features remain available) but breaks L2 and L5 entirely. This dependency should be reflected in the operational risk register.

---

## 8. Combined Portfolio Impact

| Macro condition | Net Δ in book contribution |
|---|---|
| Baseline (0pp) | **+₹17.85 cr** |
| +2pp stress | **+₹18.62 cr** |
| +3pp stress | **+₹19.01 cr** |

**Volume change:** -₹58.83 cr (~4.2% of book)

**Key insight — monotonic stress improvement:** Portfolio gain *increases* with stress severity. This is rare and powerful — most policy recommendations weaken under stress; this one strengthens because the three defensive levers (avoiding losses) overpower the one offensive lever (taking risks). The book is reshaped to be more resilient, not just more profitable.

**Per-lever breakdown:**

| Lever | Baseline | +2pp | +3pp | Volume Δ |
|---|---|---|---|---|
| L1: Cease BNPL Digital+DSA | +₹1.08 | +₹1.18 | +₹1.24 | -₹12.95 cr |
| L2: Tighten Personal D-E | +₹2.33 | +₹2.45 | +₹2.51 | -₹9.39 cr |
| L4: Grow SME A-B (10%) | +₹1.47 | +₹0.69 | +₹0.30 | +₹51.83 cr |
| L5: Contain SME C-D-E | +₹12.97 | +₹14.30 | +₹14.96 | -₹88.32 cr |
| **Combined** | **+₹17.85** | **+₹18.62** | **+₹19.01** | **-₹58.83 cr**|

---

## 9. Implementation Roadmap

| Phase | Lever | Owner | Effort | Notes |
|---|---|---|---|---|
| Month 1 | Cease BNPL Digital/DSA | BNPL Marketing | Low | Turn off ad spend, freeze DSA acquisition pipeline. Existing book runs off naturally. |
| Month 1-2 | Tighten Personal D-E | Personal Underwriting | Medium | Code the decline rule in the underwriting system; update applicant-facing decline reasons. |
| Month 2-3 | Contain SME C-D-E | SME Underwriting | Medium | Code the decline rule; brief sales/BDs on the new policy and expected decline volumes. |
| Month 3-6 | Grow SME A-B | SME Sales + BD | High | Sales ramp, partner activation, marketing investment. Takes a full quarter to reach 10% growth. |

The tightening/ceasing levers (L1, L2, L5) take effect immediately on new applications. The growth lever (L4) takes a quarter to fully ramp due to sales cycle length in SME.

---

## 10. Risks and Mitigations

**1. Volume optics.** Total origination shrinks ~₹58.8 cr (~4.2% of book). Mitigation: internally reframe as "reshape," not "shrink." Highlight per-loan economic improvement and the fact that L4's growth offsets ~47% of the shrink.

**2. Channel competitor risk.** Declining more applications may shift volume to competitors. Mitigation: monitor portfolio-level approval rate; if drop is sharper than expected, calibrate cutoff downward (e.g., 0.55 instead of 0.6) to recover some volume.

**3. Stress assumption.** +2pp / +3pp stress (A003) is the engagement-mandated band. If actual stress is *worse*, the defensive levers gain MORE; if *milder*, the recommendation still wins by ~₹17.85 cr at baseline. Tested in detail in WS H.

**4. Operational implementation.** Underwriting rule changes require ~2 weeks of engineering. No new data sources needed — `cashflow_consistency_mean` is already derived from bank statements pulled at application.

**5. Behavioural response from sales.** SME BDs may push back on declining 9% of SME C-D-E flow. Mitigation: pair the lever with the Grow SME A-B target — sales has a positive growth lever to compensate for the declines.

**6. Cutoff calibration drift.** The 0.6 threshold was derived empirically from current data. As applicant mix changes, the threshold may need updating. Recommend quarterly review of the cutoff against fresh segment loss-making rates.

---

## 11. Sensitivity Commentary

Headline recommendation tested against:

- Extended macro stress (0pp to +5pp): combined net Δ rises monotonically; robust beyond engagement-mandated +3pp band.
- **LGD ±10pp (flexed around the locked BNPL 40% / Personal 65% / SME 75%):** Combined net Δ varies by only ≈₹0.36 cr across the band (+₹18.83 cr to +₹19.19 cr at +3pp per WS H `lgd_sensitivity.csv`, re-run around the true locked LGDs). Recommendation effectively LGD-insensitive. *(See §12 reconciliation note.)*
- **CoF ±1pp:** Combined net Δ moves less than ₹0.5 cr. Insensitive. *(Locked data-generating Cost of Funds is 11%, not 9% — see §12.)*
- **Macro stress 0pp to +3pp:** Combined net Δ monotonically improves from +₹17.85 to +₹19.01 cr. Robust under stress.
- **Cashflow cutoff 0.5 to 0.7:** At 0.5 the lever shrinks (~₹8 cr combined defensive impact); at 0.7 it grows but declines too many marginal loans. **0.60 is the empirical optimum for L2 (Personal D-E) only; L5 (SME C-D-E) uses 0.65** (where the SME loss-making slice cleanly ends — see WS H §4).
- **L4 growth rate 5% to 20%:** Net Δ scales linearly at baseline but disproportionately under stress (more new loans = more stressed losses). 10% is conservative; 20% would deliver +₹2.9 cr at baseline but only +₹0.6 cr at +3pp.

Full sensitivity matrix in WS H.

---

## 12. Appendix: Assumptions Used

| ID | Assumption | Value | Source |
|---|---|---|---|
| A001 | Cost of Funds | 11.0% | Locked data-generating value (`config.py` `COST_OF_FUNDS_ANNUAL`, A-050) |
| A002 | LGD — BNPL | 40% | Locked data-generating value (`config.py` `LGD_BY_PRODUCT`, A-044) |
| A002b | LGD — Personal | 65% | Locked data-generating value (`config.py` `LGD_BY_PRODUCT`, A-044) |
| A002c | LGD — SME | 75% | Locked data-generating value (`config.py` `LGD_BY_PRODUCT`, A-044) |
| A003 | Macro stress band | +2pp / +3pp default rate | Engagement brief mandate |
| A004 | Operating cost | 1.5% of ticket | NBFC opex/AUM benchmark, typical 2-4% annual → ~1.5% per loan at typical tenures |
| A005 | CAC by channel | Per Workstream D output | Internal channel economics analysis |

> **⚠️ ASSUMPTION RECONCILIATION (corrected).** Earlier drafts of this appendix listed LGD 60% (retail)
> / 50% (SME) and Cost of Funds 9.0%. Those values do **not** match the locked, data-generating
> assumptions actually used to produce `value_proxy` in `loan_frame.parquet`. The authoritative,
> data-generating values (`config.py`, Assumptions A-044 / A-050) are:
>
> - **LGD by product:** BNPL **40%**, Personal **65%**, SME **75%** (`LGD_BY_PRODUCT`, A-044)
> - **Cost of Funds:** **11%** annual (`COST_OF_FUNDS_ANNUAL`, A-050)
>
> The appendix above has been corrected to these values. **✅ RESOLVED — WS H re-run complete.** The
> WS H LGD sensitivity (§3) and the macro-stress tables have been re-run jointly flexing all three
> product LGDs around the true locked values (BNPL 40% / Personal 65% / SME 75%) at L5 cutoff 0.65.
> Because LGD affects only the marginal stress loss (baselines are `value_proxy`-based), the **baseline
> headline of +₹17.85 cr is unchanged**, while the **+3pp headline moves from +₹18.80 cr to +₹19.01 cr**
> (and +2pp from 18.48 to 18.62); these corrected figures are now used consistently across WS G, WS H,
> and this document. The LGD-robustness claim holds: the swing across ±10pp is ≈₹0.36 cr (~1.9%).

---

## 13. Open Items / Pending Inputs

- **WS F EWS thresholds** — Section 7 to be quantified once Workstream F completes calibration. Expected ~mid-Week 2.
- **WS H sensitivity tables** — full sensitivity matrix referenced in Section 11. Currently summarized; full tables in WS H output.
- **Implementation owner sign-offs** — each lever's named owner needs to acknowledge the implementation effort and timeline before final lock.

---


