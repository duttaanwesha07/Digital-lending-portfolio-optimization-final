# Workstream E — Step 6: Method-Comparison Memo
### Evidence pack for the joint selection (Data Lead → Risk Consultant)

**Purpose.** Put the three candidate segmentations on identical footing and score them on the four selection criteria, so the pick is made on evidence. The **decision is the Risk Consultant's** (Step 6 lead); this memo is the input.

**Same-basis note.** All default rates below are on the **seasoned basis (months-on-book ≥ 3)** — the only basis on which "ever 90+ DPD" is observable. This re-states the heuristic baseline on the same footing as the tree and clustering, so the numbers compare like-for-like. (One consequence worth flagging: BNPL's default rate rises from 10.0% on all loans to **14.8% seasoned** — on the honest basis it is the riskiest product, not the safest-looking.)

---

## The scorecard

| Method | Segments (effective) | Risk separation η² | Value separation η² | Risk spread | Value spread | Interpretability | Metric CV ↓ | Membership ARI ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Heuristic baseline | 6 (6) | 0.025 | **0.026** | 11.7 pp | ₹41,867 | **High — plain rules** | 0.049 | **1.000** |
| Decision tree | 6 (6) | 0.117 | 0.008 | 40.4 pp | ₹24,350 | **High — readable splits** | 0.044 | 0.799 |
| Clustering | 4 (**3**) | **0.339** | **0.068** | 62.6 pp | ₹87,011 | Low — needs profiling (silhouette ~0.11) | 0.035 | 0.993 |

*η² = share of variance the segmentation explains (higher = better separated). Effective segments = total minus near-duplicates. CV = bootstrap variability of segment default rates (lower = steadier). ARI = membership agreement when the method is re-fit on bootstrap samples (higher = more stable).*

---

## How to read each method

**Heuristic baseline (product × grade).** The most interpretable and perfectly stable (rules don't move, ARI = 1.0), and it separates **value** reasonably well because product structure captures the SME/Personal/BNPL economics. Its weakness is **risk separation** (η² 0.025, only an 11.7 pp spread): by lumping BNPL into one "all grades" segment and using broad grade bands, it buries the dangerous pockets. It is the trustworthy backbone, but blunt on risk.

**Decision tree (grade + APR, origination-only).** The strongest **risk** lens of the three that is *actionable at underwriting* (η² 0.117, 40.4 pp spread), and it isolates the high-APR danger pocket the baseline hides — the >32.4% APR segment defaults at 42.5% and destroys value. But it has the **weakest value separation** (η² 0.008): optimised for default, its leaves barely differentiate profit. Membership is only moderately stable (ARI 0.799 — the APR thresholds shift a little across resamples).

**Clustering (behaviour + profile).** On paper it separates both risk and value best (η² 0.339 / 0.068). But three caveats gut its case as a *standalone* segmentation: (1) only **3 effective segments** — C1 and C2 are near-identical twins; (2) **low interpretability** (silhouette ~0.11, needs heavy profiling to name); (3) its separation rests almost entirely on **C4, a post-origination behavioural-distress cluster** (8× payment bounces, 7× spending shocks, 66% default). That signal is real and valuable — but it is *not knowable at origination*, so it can't be a policy lever at underwriting. It belongs to **Workstream F (early warning)**, not to a static segmentation.

---

## The finding: no method dominates — they're complementary

- The **baseline separates value** (product economics) but is blunt on risk.
- The **tree separates risk** (the APR danger cut) but is blunt on value.
- The **clustering** mainly surfaces a **behavioural early-warning signal** that points downstream to F.

This is the textbook case the Gate-3 guidance anticipates: prefer the interpretable methods, and don't adopt a statistically-tidy clustering that fails the business test.

---

## Provisional recommendation (Data Lead's read — your call to confirm)

**A documented hybrid, not a single method:**

1. **Backbone = product × risk-grade** (from the baseline) — it is interpretable, stable, and separates value.
2. **Sharpen risk with the tree's APR cut** — break out the high-APR / low-grade danger pocket (default 42.5%, value-destroying) that the baseline buries. This adds the one split that most improves risk separation while staying readable.
3. **Route clustering's C4 to Workstream F** — carry the behavioural-distress signal forward as an early-warning input, not as a static E segment.

That hybrid would land at roughly **5–6 named segments** that are separated on *both* risk and value, each nameable in a line and each tied to a distinct lever (grow SME-prime, nurture Personal-prime, re-price/contain high-APR subprime, watch the behavioural-distress group via F). It is interpretable and rule-stable.

**This feeds Step 7–8 (yours):** once you confirm the hybrid (or an alternative), you draw the risk×value matrix and name + hook each segment. **Then Step 9 (mine):** I run full stability + significance on the chosen set for Gate 3.

*Files alongside this memo: `comparison_metrics.csv` (every segment, all three methods, same basis), `method_scorecard.csv` (the table above), `stability_preview.csv` (CV + ARI).*
