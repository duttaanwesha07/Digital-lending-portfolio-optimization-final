#!/usr/bin/env python3
"""
Workstream E - Segmentation | Step 5: Unsupervised clustering
=============================================================
Data Lead track (Part 1).

Tests whether the data's NATURAL structure yields segments the rule-based
baseline (Step 3) and the risk tree (Step 4) miss. Deliberately uses the lens
those two left out: customer PROFILE + stable BEHAVIOUR + product HOLDINGS.

Design:
  * Grain = CUSTOMER (behaviour and value live at customer level). Each customer's
    loans then INHERIT the cluster, so the result lines up with the loan-grain
    baseline and tree for the Step 6 comparison.
  * Method = k-prototypes (Huang) - correct for mixed categorical+numeric data.
    Numerics standardised first. k swept over 4-6, chosen by silhouette on a
    standardised+one-hot embedding (subsample, for speed); final fit on all rows.
  * Features = profile + behaviour + holdings ONLY. No outcomes (CLV, default),
    no pricing/economics (APR, value), no z. Those are used only to PROFILE.
  * value_proxy frozen; characterisation only.

Run from the wsE root:   python methods/cluster_segments.py
Reads:   wsE/out/frames/{customer_frame,loan_frame}.parquet
Writes:  wsE/out/methods/  (customer + loan assignments, metrics, profiles, selection, manifest)
"""

from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from kmodes.kprototypes import KPrototypes

# ======================================================================
# CONFIG
# ======================================================================
SEED = 20260603
ROOT   = Path(__file__).resolve().parent.parent
FRAMES = ROOT / "out" / "frames"
OUTDIR = ROOT / "out" / "methods"
CRORE  = 10_000_000
SEASONED_MOB_MIN = 3                       # same honest basis as the Step-4 tree
KS = [4, 5, 6]
SWEEP_SUBSAMPLE = 6000
SIL_SUBSAMPLE   = 5000

# who they ARE + how they BEHAVE + what they HOLD  (no outcomes, no economics, no z)
NUMERIC_FEATURES = ["age", "cashflow_consistency_mean", "balance_volatility_mean",
                    "credit_utilisation_mean", "app_engagement_mean",
                    "bureau_inquiry_velocity_mean", "nach_bounce_total",
                    "spending_shock_rate", "n_loans", "n_products_held"]
CATEGORICAL_FEATURES = ["region_type", "income_proxy_band", "employment_type",
                        "credit_quality_indicator", "acquisition_channel",
                        "primary_product", "first_time_borrower"]
FORBIDDEN = {"true_latent_risk", "z", "clv", "any_default", "n_defaults", "n_writeoffs",
             "value_proxy", "default_flag", "ever_90plus", "worst_delq_rank_ever",
             "clv_to_cac", "net_value_positive", "total_originated", "mean_apr"}


def _fail(msg):
    print(f"\n  [X] CHECK FAILED: {msg}"); sys.exit(1)


# ======================================================================
# feature construction
# ======================================================================
def build_features(cf):
    leaked = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c.lower() in FORBIDDEN]
    if leaked: _fail(f"forbidden feature in clustering inputs: {leaked}")
    Xn = StandardScaler().fit_transform(cf[NUMERIC_FEATURES].astype(float))
    Xc = cf[CATEGORICAL_FEATURES].astype(str).values
    X = np.hstack([Xn, Xc]).astype(object)             # for k-prototypes
    cat_idx = list(range(len(NUMERIC_FEATURES), len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)))
    # embedding for silhouette: standardised numerics + one-hot categoricals
    onehot = pd.get_dummies(cf[CATEGORICAL_FEATURES].astype(str)).values
    emb = np.hstack([Xn, onehot]).astype(float)
    return X, cat_idx, emb


def fit_kproto(X, cat_idx, k, n_init):
    return KPrototypes(n_clusters=k, init="Huang", n_init=n_init, max_iter=20,
                       random_state=SEED, n_jobs=1).fit(X, categorical=cat_idx)


# ======================================================================
# profiling
# ======================================================================
def cluster_profiles(cf, lf, cluster_col):
    lf = lf.copy()
    lf["seasoned"] = (lf.months_on_book >= SEASONED_MOB_MIN).astype(int)
    total_cust, total_loans, total_orig = len(cf), len(lf), lf.ticket_size.sum()
    rows, texts = [], []
    for c, d in cf.groupby(cluster_col, sort=True):
        ld = lf[lf.customer_id.isin(d.customer_id)]
        s = ld[ld.seasoned == 1]
        def top(col):
            vc = d[col].astype(str).value_counts(normalize=True)
            return vc.index[0], round(vc.iloc[0] * 100)
        reg, reg_p = top("region_type"); inc, _ = top("income_proxy_band")
        emp, _ = top("employment_type"); prod, prod_p = top("primary_product")
        rows.append(dict(
            cluster=c, n_customers=len(d), pct_customers=round(100 * len(d) / total_cust, 1),
            n_loans=len(ld), pct_of_book=round(100 * len(ld) / total_loans, 1),
            dominant_region=reg, dominant_income=inc, dominant_employment=emp,
            primary_product=prod, primary_product_share=prod_p,
            mean_age=round(d.age.mean(), 1),
            mean_cashflow_consistency=round(d.cashflow_consistency_mean.mean(), 3),
            mean_credit_utilisation=round(d.credit_utilisation_mean.mean(), 1),
            spending_shock_rate=round(d.spending_shock_rate.mean(), 3),
            cust_default_rate_pct=round(100 * d.any_default.mean(), 2),
            loan_default_rate_seasoned_pct=round(100 * s.default_flag.mean(), 2) if len(s) else np.nan,
            mean_clv_inr=round(d.clv.mean()),
            mean_value_proxy_inr=round(ld.value_proxy.mean()),
            total_value_proxy_cr=round(ld.value_proxy.sum() / CRORE, 2),
            total_originated_cr=round(ld.ticket_size.sum() / CRORE, 1),
            pct_loss_making_loans=round(100 * (ld.value_proxy < 0).mean(), 1),
        ))
        texts.append(f"{c}  ({len(d):,} customers, {round(100*len(d)/total_cust,1)}% | "
                     f"{len(ld):,} loans)\n"
                     f"     WHO   : mostly {reg} ({reg_p}%), {inc}-income, {emp}; "
                     f"primarily {prod} ({prod_p}%)\n"
                     f"     BEHAVE: cashflow consistency {round(d.cashflow_consistency_mean.mean(),2)}, "
                     f"utilisation {round(d.credit_utilisation_mean.mean(),1)}%, "
                     f"shock rate {round(d.spending_shock_rate.mean(),2)}\n"
                     f"     RISK  : seasoned loan default "
                     f"{round(100*s.default_flag.mean(),2) if len(s) else float('nan')}% "
                     f"(customer ever-default {round(100*d.any_default.mean(),1)}%)\n"
                     f"     VALUE : mean CLV \u20b9{round(d.clv.mean()):,} | "
                     f"{round(100*(ld.value_proxy<0).mean(),1)}% loss-making loans\n")
    return pd.DataFrame(rows), texts


# ======================================================================
# main
# ======================================================================
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cf = pd.read_parquet(FRAMES / "customer_frame.parquet")
    lf = pd.read_parquet(FRAMES / "loan_frame.parquet")
    for nm, df in [("customer", cf), ("loan", lf)]:
        if [c for c in df.columns if c.lower() in ("true_latent_risk", "z")]:
            _fail(f"z present in {nm}_frame")
    print(f"Loaded {len(cf):,} customers / {len(lf):,} loans")

    X, cat_idx, emb = build_features(cf)
    print(f"Clustering features: {len(NUMERIC_FEATURES)} numeric + {len(CATEGORICAL_FEATURES)} categorical "
          f"(profile + behaviour + holdings; no outcomes/economics/z)")

    # ---- k sweep on a subsample ----
    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(cf), size=min(SWEEP_SUBSAMPLE, len(cf)), replace=False)
    print(f"\nSweeping k over {KS} on a {len(sub):,}-customer subsample:")
    sweep = []
    for k in KS:
        km = fit_kproto(X[sub], cat_idx, k, n_init=1)
        sil = silhouette_score(emb[sub], km.labels_, sample_size=min(SIL_SUBSAMPLE, len(sub)),
                               random_state=SEED)
        sweep.append(dict(k=k, silhouette=round(sil, 4), cost=round(km.cost_)))
        print(f"   k={k}: silhouette={sil:.4f}  cost={km.cost_:.0f}")
    sweep_df = pd.DataFrame(sweep)
    best_k = int(sweep_df.loc[sweep_df.silhouette.idxmax(), "k"])
    print(f"   -> chosen k = {best_k} (highest silhouette)")

    # ---- final fit on all customers ----
    print(f"\nFinal k-prototypes fit (k={best_k}) on all {len(cf):,} customers ...")
    final = fit_kproto(X, cat_idx, best_k, n_init=2)
    cf = cf.copy()
    cf["cluster_id"] = final.labels_

    # order clusters by seasoned loan default rate -> label C1 (safest) .. Ck
    lf2 = lf.merge(cf[["customer_id", "cluster_id"]], on="customer_id", how="left")
    lf2["seasoned"] = (lf2.months_on_book >= SEASONED_MOB_MIN).astype(int)
    sdef = (lf2[lf2.seasoned == 1].groupby("cluster_id").default_flag.mean())
    order = list(sdef.sort_values().index)
    relabel = {cid: f"C{i+1}" for i, cid in enumerate(order)}
    cf["cluster"] = cf.cluster_id.map(relabel)
    lf2["cluster"] = lf2.cluster_id.map(relabel)

    sil_final = silhouette_score(emb, final.labels_, sample_size=SIL_SUBSAMPLE, random_state=SEED)
    metrics, texts = cluster_profiles(cf, lf2, "cluster")

    # ---- cross-checks ----
    print("\nCross-checks")
    if cf.cluster.isna().any():  _fail("unassigned customers")
    if lf2.cluster.isna().any(): _fail("unassigned loans")
    if len(cf) != 40000 or len(lf2) != 49600: _fail("row-count drift")
    print(f"  [ok] all {len(cf):,} customers + {len(lf2):,} loans assigned to {best_k} clusters")
    rec = round(lf2.value_proxy.sum() / CRORE, 2)
    if abs(metrics.total_value_proxy_cr.sum() - rec) > 0.05: _fail("value does not reconcile")
    print(f"  [ok] value reconciles: {metrics.total_value_proxy_cr.sum():.2f} cr (~\u20b9473.6M)")
    dspread = metrics.loan_default_rate_seasoned_pct.max() - metrics.loan_default_rate_seasoned_pct.min()
    vspread = metrics.mean_clv_inr.max() - metrics.mean_clv_inr.min()
    print(f"  [ok] separation: seasoned default spread {dspread:.1f}pp | mean-CLV spread \u20b9{vspread:,.0f}")
    flat = metrics[(metrics.loan_default_rate_seasoned_pct.rank().between(2, best_k-1)) &
                   (metrics.pct_loss_making_loans.between(metrics.pct_loss_making_loans.min(),
                                                          metrics.pct_loss_making_loans.min()))]
    print(f"  [ok] final silhouette (all rows, sampled) = {sil_final:.4f}")
    print("  [ok] no outcome / economics / z field used as a clustering input")

    # ---- write ----
    cf[["customer_id", "cluster"]].to_csv(OUTDIR / "cluster_customer_assignments.csv", index=False)
    lf2[["loan_id", "customer_id", "cluster"]].to_csv(OUTDIR / "cluster_loan_assignments.csv", index=False)
    metrics.to_csv(OUTDIR / "cluster_metrics.csv", index=False)
    sweep_df.assign(chosen=lambda d: d.k == best_k).to_csv(OUTDIR / "cluster_selection.csv", index=False)
    with open(OUTDIR / "cluster_profiles.txt", "w") as f:
        f.write(f"WS-E Step 5 - k-prototypes clusters (k={best_k}); customer-grain, loans inherit.\n")
        f.write("Default rates on seasoned loans (MOB>=3). Provisional C-labels; naming is Step 8.\n")
        f.write("=" * 70 + "\n\n")
        for t in texts: f.write(t + "\n")
    json.dump({"step": "WS-E Step 5 - clustering", "seed": SEED, "method": "k-prototypes (Huang)",
               "grain": "customer (loans inherit)", "k_chosen": best_k,
               "k_sweep": sweep, "final_silhouette": round(float(sil_final), 4),
               "features": {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES},
               "default_rate_basis": f"seasoned loans (MOB>={SEASONED_MOB_MIN})",
               "z_excluded": True, "outcomes_excluded": True, "value_proxy": "frozen; profiling only"},
              open(OUTDIR / "cluster_manifest.json", "w"), indent=2)

    pd.set_option("display.width", 240); pd.set_option("display.max_columns", None)
    print("\nClusters (provisional C-labels):")
    print(metrics[["cluster", "n_customers", "pct_customers", "primary_product",
                   "dominant_income", "loan_default_rate_seasoned_pct", "mean_clv_inr",
                   "pct_loss_making_loans"]].to_string(index=False))
    print(f"\nWrote -> {OUTDIR}\nStep 5 complete. All checks passed.")


if __name__ == "__main__":
    main()
