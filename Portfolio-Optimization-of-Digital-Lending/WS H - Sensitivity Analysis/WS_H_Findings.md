# Workstream H — Sensitivity Analysis Findings

**Prepared by:** Anwesha
**Date:** 12 June 2026
**Status:** Final
**Inputs:** WS G scenarios and lever computations.

---

## 1. Summary

Workstream H stress-tests the WS G recommendation against four independent assumption ranges to assess robustness. Three of the four sensitivities confirm the recommendation; the fourth (cashflow cutoff) revealed a sub-optimal cutoff for Lever 5 (Contain SME C-D-E) that has been corrected.

**Headline findings:**

1. **Extended macro stress (0pp to +5pp):** Combined Net Δ rises monotonically from +₹17.85 cr (0pp) to +₹19.78 cr (+5pp). Recommendation strengthens under stress; no break-point identified within plausible range. Current headline is **+₹17.85 cr baseline / +₹19.01 cr at +3pp**.

2. **LGD ±10pp:** Combined Net Δ varies by only ~₹0.36 cr across the full ±10pp band (+₹18.83 to +₹19.19 cr at +3pp stress, ~1.9% of the headline). Recommendation is effectively LGD-insensitive due to offsetting effects between defensive and offensive levers. Re-run around the true locked LGDs (BNPL 40% / Personal 65% / SME 75%, A-044).

3. **Cashflow cutoff (0.50 to 0.70):** Empirical optimum confirmed for L2 (Personal D-E) at 0.60. **Empirical optimum for L5 (SME C-D-E) is 0.65, not 0.60 as originally specified — see Section 4 for details.** L5 cutoff has been updated.

4. **L4 growth rate (5% to 20%):** Both baseline gain and stress loss scale linearly with growth rate; all of 5–20% stay positive at baseline, and 10% (the recommended target) stays positive (+₹0.30 cr) even at +3pp. See Section 5.

---

## 2. Extended Macro Stress

**Method:** Compute combined portfolio Net Δ at default-rate stress levels from 0pp to +5pp.
Computed at L5 cutoff = 0.65 and the locked data-generating LGDs (BNPL 40% / Personal 65% / SME 75%, A-044).

**Result (Net Δ in ₹ cr):**

| Stress (pp) | L1 | L2 | L4 | L5 | Combined |
|---|---|---|---|---|---|
| 0 | 1.08 | 2.33 | 1.47 | 12.97 | 17.85 |
| 1 | 1.13 | 2.39 | 1.08 | 13.64 | 18.24 |
| 2 | 1.18 | 2.45 | 0.69 | 14.30 | 18.62 |
| 3 | 1.24 | 2.51 | 0.30 | 14.96 | 19.01 |
| 4 | 1.29 | 2.57 | -0.09 | 15.62 | 19.40 |
| 5 | 1.34 | 2.63 | -0.48 | 16.29 | 19.78 |

**Interpretation:**
- The three defensive levers (L1, L2, L5) each gain impact as stress rises — the loans they decline would have caused even larger losses under worse macro conditions.
- The offensive lever (L4 — Grow SME A-B) weakens with stress because new loans take stressed losses. L4 alone goes negative at approximately +3.8pp stress.
- Combined portfolio impact rises monotonically. Defensive levers' gains exceed L4's losses at every stress level tested.
- No break-point is reached within the plausible macro range (Indian retail credit has not seen stress beyond +4pp since the 1991 BoP crisis).

**Implication for the recommendation:** robust to macro stress significantly beyond the engagement-mandated +3pp band.

---

## 3. LGD Sensitivity

**Method:** Hold +3pp macro stress constant. Flex retail LGD (50-70%) and SME LGD (40-60%) jointly.

> **✅ RE-RUN COMPLETE — flexed around the locked data-generating LGDs.** `value_proxy` in
> `loan_frame.parquet` was generated with the locked assumptions in `config.py` (A-044):
> **LGD = BNPL 40% / Personal 65% / SME 75%**, and Cost of Funds **11%** (A-050). An earlier draft of
> this analysis flexed around a 60% retail / 50% SME centre that did **not** match the data; it has
> been re-run jointly flexing all three product LGDs ±10pp around their true locked values. L5 cutoff
> = 0.65. Note: LGD affects only the marginal stress loss (baselines are `value_proxy`-based and
> LGD-independent), so the portfolio remains LGD-insensitive.

**Result (Combined Net Δ at +3pp stress, in ₹ cr):**

| Scenario | BNPL LGD | Personal LGD | SME LGD | L1 | L2 | L4 | L5 | Combined |
|---|---|---|---|---|---|---|---|---|
| LGD −10pp (optimistic) | 30% | 55% | 65% | 1.20 | 2.48 | 0.46 | 14.70 | 18.83 |
| LGD −5pp | 35% | 60% | 70% | 1.22 | 2.50 | 0.38 | 14.83 | 18.92 |
| Locked (baseline) | 40% | 65% | 75% | 1.24 | 2.51 | 0.30 | 14.96 | 19.01 |
| LGD +5pp | 45% | 70% | 80% | 1.26 | 2.53 | 0.22 | 15.09 | 19.10 |
| LGD +10pp (pessimistic) | 50% | 75% | 85% | 1.27 | 2.54 | 0.15 | 15.23 | 19.19 |

**Interpretation:**

The recommendation is **essentially LGD-insensitive at the portfolio level**. Combined Net Δ varies by only ₹0.36 cr (+₹18.83 to +₹19.19) across the full ±10pp LGD band — a ~1.9% variation on a +₹19.01 cr recommendation.

**Why this happens:** LGD has *opposing* effects on different lever types.

- For cease/tighten levers (L1, L2, L5): higher LGD increases the additional losses we *avoid* under stress, so the lever gains impact.
- For the grow lever (L4): higher LGD increases the losses we *take* on the new loans under stress, so the lever loses impact.

At the portfolio level, the defensive gains and offensive losses roughly cancel. This is a structural property of the portfolio's lever mix, not a coincidence.

**Implication:** the recommendation does not hinge on getting the LGD assumption exactly right. Even across the full ±10pp band around the locked LGDs (BNPL 40% / Personal 65% / SME 75%), the headline number moves by less than ₹0.4 cr.

---

## 4. Cashflow Cutoff Sensitivity — Discovery and Correction

**Method:** Re-compute L2 (Personal D-E) and L5 (SME C-D-E) impacts at cashflow_consistency cutoffs from 0.50 to 0.70 in 0.05 increments.

**Result (Net Δ in ₹ cr):**

| Cutoff | L2 loans | L2 baseline | L2 at +3pp | L5 loans | L5 baseline | L5 at +3pp |
|---|---|---|---|---|---|---|
| 0.50 | 210 | 0.82 | 0.87 | 151 | 4.04 | 4.35 |
| 0.55 | 395 | 1.50 | 1.59 | 265 | 6.75 | 7.30 |
| 0.60 | 767 | 2.33 | 2.51 | 498 | 10.61 | 11.63 |
| 0.65 | 1,409 | 2.42 | 2.75 | 950 | **12.97** | **14.96** |
| 0.70 | 2,935 | 0.73 | 1.43 | 2,035 | 6.05 | 10.28 |

*+3pp columns computed at the locked LGDs (Personal 65% / SME 75%, A-044); baselines are
`value_proxy`-based and LGD-independent.*

**Finding for L2 (Personal D-E):** Impact at 0.60 is +₹2.33 cr; at 0.65 it is +₹2.42 cr. The 0.05 cr improvement is within noise. The 0.65-0.70 range drops sharply (+₹2.42 → +₹0.73 at baseline). **0.60 is confirmed as the practical optimum for L2.**

**Finding for L5 (SME C-D-E):** Impact at 0.60 is +₹10.61 cr; at 0.65 it jumps to +₹12.97 cr — a +₹2.36 cr improvement (22% gain). At 0.70, impact collapses to +₹6.05 cr. **0.65 is the empirical optimum for L5; 0.60 was sub-optimal.**

**Why L2 and L5 differ:**

SME borrowers and Personal borrowers have different cashflow profiles. Within Personal D-E, the loss-making slice ends cleanly near cashflow_consistency = 0.60 — the sign of mean value_proxy flips between the 0.5-0.6 band and the 0.6-0.7 band. Within SME C-D-E, the loss-making slice extends slightly further. The 452 additional SME loans declined at cutoff 0.65 (vs 0.60) average **−₹52,391 in value_proxy each** (recomputed from `loan_frame.parquet`; the previously reported −₹5,221 understated the per-loan loss ~10×) — they are real loss-makers, just on the right side of the original threshold. The larger per-loan loss *strengthens*, not weakens, the case for moving the cutoff to 0.65. Beyond 0.65, the loans become profitable (mean +₹63K each in the 0.65-0.70 range), which is why 0.70 collapses the lever impact.

**Action taken:** Lever 5 cutoff updated from 0.60 to 0.65. Workstream G scenarios and Workstream I recommendation document updated accordingly.

**Impact of the correction:**

| Metric | Original (0.60) | Updated (0.65) |
|---|---|---|
| L5 loans declined | 498 | 950 |
| L5 origination given up | ₹45.64 cr | ₹88.32 cr |
| L5 Net Δ at baseline | +₹10.61 cr | +₹12.97 cr |
| L5 Net Δ at +3pp | +₹11.29 cr | +₹14.96 cr |
| Combined Net Δ at baseline | +₹15.48 cr | **+₹17.85 cr** |
| Combined Net Δ at +3pp | +₹15.79 cr | **+₹19.01 cr** |
| Total origination Δ | -₹16.15 cr | -₹58.83 cr |

The correction adds ~₹2.4 cr to the headline gain at baseline (and ~₹3.2 cr at +3pp, once the +3pp
figures are also recomputed at the true locked LGDs). The volume hit increases from 1% to ~4% of the
book — still manageable but more substantial. Recommendation: WS I narrative should acknowledge the
slightly larger volume reshape and frame as "trade a 4% reduction in origination for a 6%+ improvement
in book contribution." *(The "Updated (0.65)" column uses the locked data-generating LGDs — BNPL 40% /
Personal 65% / SME 75%, A-044; the "Original (0.60)" column reflects the pre-correction reported figures.)*

---

## 5. L4 Growth Rate Sensitivity

**Method:** Scale the Grow SME A-B lever from 5% to 20% growth; report Net Δ at baseline and +3pp
(SME LGD 75%, A-044).

**Result (Net Δ in ₹ cr):**

| Growth % | New loans | Origination added (cr) | Baseline | +3pp stress | Δ baseline→+3pp |
|---|---|---|---|---|---|
| 5 | 228 | 25.92 | 0.73 | 0.15 | -0.58 |
| 10 | 456 | 51.83 | 1.47 | 0.30 | -1.17 |
| 15 | 684 | 77.75 | 2.20 | 0.45 | -1.75 |
| 20 | 912 | 103.67 | 2.94 | 0.60 | -2.33 |

**Interpretation:** Baseline gain scales linearly with growth (≈+₹0.147 cr per 1% growth), while the
+3pp stress loss scales faster (the new loans take stressed losses at SME LGD 75%). All growth rates
5–20% stay **positive even at +3pp**, but the margin thins as growth rises — at 20% the +3pp gain is
only +₹0.60 cr. **10% is the conservative target adopted:** meaningful volume replacement (₹51.83 cr)
while staying comfortably positive under stress (+₹0.30 cr at +3pp). SME A-B is the only segment where
growth is safe under stress because of its high margin cushion (₹32,204 per-loan margin, 2.74% default
rate).

---

## 6. Implications Summary

WS H sensitivity analysis confirms the recommendation is:

- **Robust to severe macro stress** (positive across 0pp to +5pp; no break-point in plausible range)
- **Insensitive to LGD assumption** (less than ₹0.4 cr variation across LGD ±10pp around the locked BNPL 40% / Personal 65% / SME 75%)
- **Empirically optimized on cutoff selection** (0.60 confirmed for L2, 0.65 newly identified and adopted for L5)
- **Defensible against challenge** on each major assumption

Two material updates were triggered by WS H: the **L5 cutoff correction (0.60 → 0.65)** and the
**LGD re-run around the true locked data-generating values** (BNPL 40% / Personal 65% / SME 75%,
A-044), which moves the +3pp headline. The locked recommendation is now:

- **Headline:** Contain SME C-D-E at cashflow_consistency < **0.65** → +₹12.97 cr baseline, +₹14.96 cr at +3pp
- **Supporting 1:** Tighten Personal D-E at cashflow_consistency < 0.60 → +₹2.33 cr baseline (unchanged)
- **Supporting 2:** Cease BNPL Digital/DSA → +₹1.08 cr baseline (unchanged)
- **Offensive:** Grow SME A-B by 10% → +₹1.47 cr baseline (unchanged)
- **Combined:** **+₹17.85 cr at baseline, +₹19.01 cr at +3pp stress**

---

## 7. Methodology Notes

All sensitivities computed using the lever functions defined in `WS H_sensitivity.ipynb`:

- `compute_cease_net(target_loans, lgd, stress_pp)` — for cease/tighten levers
- `compute_grow_net(n_loans, mean_vp, mean_ticket, lgd, stress_pp)` — for grow levers

Underlying data: `loan_frame.parquet` (49,600 loans) and `step3_segment_assignments.csv`, both produced by Workstream C and consumed unchanged.

Output files in `WS H/`:
- `extended_stress.csv` — extended macro stress results
- `lgd_sensitivity.csv` — LGD flex results
- `cutoff_sensitivity.csv` — cashflow cutoff flex results
- `growth_rate_sensitivity.csv` — L4 growth rate results

*All four CSVs and this document were regenerated from `WS H_sensitivity.ipynb` at L5 cutoff = 0.65 and
the locked data-generating LGDs (BNPL 40% / Personal 65% / SME 75%, A-044). The notebook runs
end-to-end against the canonical `../WS E - Customer Segmentation/out/frames/loan_frame.parquet`.*

---

