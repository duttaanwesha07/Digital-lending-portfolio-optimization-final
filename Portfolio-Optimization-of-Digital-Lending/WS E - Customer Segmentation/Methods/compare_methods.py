#!/usr/bin/env python3
"""
Workstream E - Segmentation | Step 6: Head-to-head selection (evidence pack)
============================================================================
Data Lead track (Part 1) - feeds the JOINT selection the Risk Consultant leads.

Puts the three methods on IDENTICAL footing and scores them on the four
selection criteria, so the pick is made on evidence, not vibes:
  (i)  separation   - eta^2 (variance explained) on risk & value + spreads
  (ii) interpretability - documented qualitative tier
  (iii) actionability   - effective segment count (near-duplicates flagged)
  (iv) stability    - bootstrap metric CV + membership stability (ARI by re-fit)

Fairness fix: the heuristic baseline (Step 3) reported default on ALL loans;
the tree and clustering use the seasoned basis (MOB>=3). Here ALL three are put
on the seasoned basis so default rates compare like-for-like.

Reads:   wsE/out/frames/loan_frame.parquet
         baseline:  inputs/step3_segment_assignments.csv  (her Step 3)
         tree:      out/methods/tree_segment_assignments.csv
         cluster:   out/methods/cluster_loan_assignments.csv
Writes:  out/methods/  comparison_metrics.csv, method_scorecard.csv, stability_preview.csv
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import adjusted_rand_score
from kmodes.kprototypes import KPrototypes

SEED = 20260603
ROOT   = Path(__file__).resolve().parent.parent
FRAMES = ROOT / "out" / "frames"
METHODS = ROOT / "out" / "methods"
INPUTS = ROOT / "inputs"
CRORE = 10_000_000
SEASONED_MOB_MIN = 3
CRORE = 10_000_000

# baseline assignment file may live in inputs/ (sent by colleague)
BASELINE_CANDIDATES = [INPUTS / "step3_segment_assignments.csv",
                       ROOT / "step3_segment_assignments.csv",
                       METHODS / "step3_segment_assignments.csv"]


def _fail(m): print(f"\n  [X] {m}"); sys.exit(1)

def eta_squared(y, groups):
    y = np.asarray(y, float); grand = y.mean()
    ss_tot = ((y - grand) ** 2).sum()
    ss_bet = sum(len(y[groups == g]) * (y[groups == g].mean() - grand) ** 2
                 for g in pd.unique(groups))
    return ss_bet / ss_tot if ss_tot > 0 else np.nan

def per_segment(lf, seg):
    tot, orig = len(lf), lf.ticket_size.sum()
    out = []
    for s, d in lf.groupby(seg, sort=True):
        sd = d[d.seasoned == 1]
        out.append(dict(method=None, segment=str(s), n_loans=len(d),
                        pct_of_book=round(100*len(d)/tot, 1),
                        seasoned_default_pct=round(100*sd.default_flag.mean(), 2) if len(sd) else np.nan,
                        mean_value_inr=round(d.value_proxy.mean()),
                        total_value_cr=round(d.value_proxy.sum()/CRORE, 2),
                        pct_loss_making=round(100*(d.value_proxy < 0).mean(), 1)))
    return pd.DataFrame(out)

def near_duplicates(seg_df):
    """count segment pairs that are near-identical on BOTH risk and value."""
    n, pairs = len(seg_df), 0
    for i in range(n):
        for j in range(i+1, n):
            a, b = seg_df.iloc[i], seg_df.iloc[j]
            if (abs(a.seasoned_default_pct - b.seasoned_default_pct) < 1.0 and
                abs(a.mean_value_inr - b.mean_value_inr) < 2000):
                pairs += 1
    return pairs


# ---- tree feature build (compact, for re-fit stability) ----
GMAP = {"A":1,"B":2,"C":3,"D":4,"E":5}
TNUM = ["ticket_size","tenure_mo","interest_rate_apr","approval_tat_hrs","age","first_time_borrower","loan_sequence"]
TONEHOT = ["product_type","region_type","income_proxy_band","employment_type","acquisition_channel"]
def tree_X(lf):
    cols=[lf[c].astype(float).values.reshape(-1,1) for c in TNUM]; names=list(TNUM)
    for c in ["origination_risk_grade","credit_quality_indicator"]:
        cols.append(lf[c].astype(str).map(GMAP).values.reshape(-1,1)); names.append(c+"_ord")
    for c in TONEHOT:
        d=pd.get_dummies(lf[c].astype(str),prefix=c,prefix_sep="="); cols.append(d.values); names+=list(d.columns)
    return np.hstack(cols).astype(float), names

# ---- cluster feature build (compact, for re-fit stability) ----
CNUM=["age","cashflow_consistency_mean","balance_volatility_mean","credit_utilisation_mean",
      "app_engagement_mean","bureau_inquiry_velocity_mean","nach_bounce_total","spending_shock_rate",
      "n_loans","n_products_held"]
CCAT=["region_type","income_proxy_band","employment_type","credit_quality_indicator",
      "acquisition_channel","primary_product","first_time_borrower"]
def cluster_X(cf):
    Xn=StandardScaler().fit_transform(cf[CNUM].astype(float)); Xc=cf[CCAT].astype(str).values
    return np.hstack([Xn,Xc]).astype(object), list(range(len(CNUM),len(CNUM)+len(CCAT)))


def main():
    lf = pd.read_parquet(FRAMES / "loan_frame.parquet")
    cf = pd.read_parquet(FRAMES / "customer_frame.parquet")
    lf["seasoned"] = (lf.months_on_book >= SEASONED_MOB_MIN).astype(int)

    base_path = next((p for p in BASELINE_CANDIDATES if p.exists()), None)
    if base_path is None: _fail(f"baseline assignments not found in {[str(p) for p in BASELINE_CANDIDATES]}")
    base = pd.read_csv(base_path)[["loan_id", "segment"]].rename(columns={"segment": "seg_base"})
    tree = pd.read_csv(METHODS / "tree_segment_assignments.csv")[["loan_id","segment"]].rename(columns={"segment":"seg_tree"})
    clus = pd.read_csv(METHODS / "cluster_loan_assignments.csv")[["loan_id","cluster"]].rename(columns={"cluster":"seg_clus"})
    lf = lf.merge(base, on="loan_id").merge(tree, on="loan_id").merge(clus, on="loan_id")
    print(f"Merged all three methods onto {len(lf):,} loans (seasoned basis MOB>={SEASONED_MOB_MIN})")

    methods = {"Heuristic baseline": "seg_base", "Decision tree": "seg_tree", "Clustering": "seg_clus"}

    # ---- per-segment metrics (same basis) ----
    seg_tables = {}
    for name, col in methods.items():
        t = per_segment(lf, col); t["method"] = name; seg_tables[name] = t
    comparison = pd.concat(seg_tables.values(), ignore_index=True)[
        ["method","segment","n_loans","pct_of_book","seasoned_default_pct",
         "mean_value_inr","total_value_cr","pct_loss_making"]]

    # ---- separation + actionability ----
    seas = lf[lf.seasoned == 1]
    rows = []
    for name, col in methods.items():
        risk_eta = eta_squared(seas.default_flag.values, seas[col].values)
        val_eta  = eta_squared(lf.value_proxy.values, lf[col].values)
        st = seg_tables[name]
        rows.append(dict(method=name, n_segments=len(st),
                         risk_eta2=round(risk_eta, 4), value_eta2=round(val_eta, 4),
                         risk_spread_pp=round(st.seasoned_default_pct.max()-st.seasoned_default_pct.min(), 1),
                         value_spread_inr=round(st.mean_value_inr.max()-st.mean_value_inr.min()),
                         near_duplicate_pairs=near_duplicates(st),
                         effective_segments=len(st)-near_duplicates(st)))
    sep = pd.DataFrame(rows)

    # ---- stability: bootstrap metric CV (all three, fixed assignments) ----
    print("Bootstrap metric stability (B=200) ...")
    rng = np.random.default_rng(SEED); B = 200; n = len(lf)
    seas_idx = lf.index[lf.seasoned == 1].values
    metric_cv = {}
    for name, col in methods.items():
        segs = sorted(lf[col].unique())
        drates = {s: [] for s in segs}
        for _ in range(B):
            samp = lf.loc[rng.choice(seas_idx, size=len(seas_idx), replace=True)]
            gr = samp.groupby(col).default_flag.mean()
            for s in segs: drates[s].append(gr.get(s, np.nan))
        cvs = [np.nanstd(v)/np.nanmean(v) for v in drates.values() if np.nanmean(v) > 0]
        metric_cv[name] = round(float(np.mean(cvs)), 4)

    # ---- stability: membership via re-fit (ARI vs base assignment) ----
    print("Membership stability (re-fit ARI) ...")
    ari = {}
    # baseline = deterministic rules -> membership invariant by construction
    ari["Heuristic baseline"] = 1.0
    # tree: 25 bootstrap re-fits on seasoned loans, predict full, ARI vs base tree labels
    Xtree, _ = tree_X(lf); ybase = lf.seg_tree.values
    seas_mask = (lf.seasoned == 1).values
    aris = []
    for _ in range(25):
        bs = rng.choice(np.where(seas_mask)[0], size=seas_mask.sum(), replace=True)
        clf = DecisionTreeClassifier(max_leaf_nodes=6, min_samples_leaf=1000, max_depth=4,
                                     class_weight="balanced", random_state=SEED).fit(Xtree[bs], lf.default_flag.values[bs])
        aris.append(adjusted_rand_score(ybase, clf.apply(Xtree)))
    ari["Decision tree"] = round(float(np.mean(aris)), 3)
    # clustering: 3 bootstrap re-fits (reduced params), predict full, ARI vs base
    Xc, cat_idx = cluster_X(cf); cbase = cf.merge(
        pd.read_csv(METHODS/"cluster_customer_assignments.csv"), on="customer_id").cluster.values
    caris = []
    for _ in range(3):
        bs = rng.choice(len(cf), size=len(cf), replace=True)
        kp = KPrototypes(n_clusters=4, init="Huang", n_init=1, max_iter=10, random_state=SEED, n_jobs=1).fit(Xc[bs], categorical=cat_idx)
        caris.append(adjusted_rand_score(cbase, kp.predict(Xc, categorical=cat_idx)))
    ari["Clustering"] = round(float(np.mean(caris)), 3)

    # ---- scorecard ----
    interp = {"Heuristic baseline": "High (plain rules)", "Decision tree": "High (readable splits)",
              "Clustering": "Low (needs profiling; silhouette ~0.11)"}
    sc = sep.copy()
    sc["interpretability"] = sc.method.map(interp)
    sc["metric_cv_lower_better"] = sc.method.map(metric_cv)
    sc["membership_ARI_higher_better"] = sc.method.map(ari)

    comparison.to_csv(METHODS / "comparison_metrics.csv", index=False)
    sc.to_csv(METHODS / "method_scorecard.csv", index=False)
    pd.DataFrame([{"method": m, "metric_cv": metric_cv[m], "membership_ARI": ari[m]} for m in methods]).to_csv(
        METHODS / "stability_preview.csv", index=False)

    pd.set_option("display.width", 240); pd.set_option("display.max_columns", None)
    print("\n================ SCORECARD ================")
    print(sc[["method","n_segments","effective_segments","risk_eta2","value_eta2",
              "risk_spread_pp","value_spread_inr","near_duplicate_pairs","interpretability",
              "metric_cv_lower_better","membership_ARI_higher_better"]].to_string(index=False))
    print("\nBaseline on the SAME seasoned basis (default rates now comparable):")
    print(seg_tables["Heuristic baseline"][["segment","n_loans","seasoned_default_pct","mean_value_inr","pct_loss_making"]].to_string(index=False))
    print(f"\nWrote -> {METHODS}\nStep 6 evidence pack complete.")


if __name__ == "__main__":
    main()
