#!/usr/bin/env python3
"""
Workstream E - Segmentation | Step 4: Tree-based segmentation  (v2: seasoned)
============================================================================
Data Lead track (Part 1).

v2 fixes the censoring artifact found in v1. default_flag = "ever 90+ DPD
(sticky)", which is MECHANICALLY impossible before a loan has been on book ~3
months. v1 therefore produced two leaves of ultra-short BNPL with a fake 0%
default rate. The data confirms it: loans at MOB 0-2 show exactly 0.00% default
across 9,133 loans, and default only "switches on" at MOB 3.

THE FIX (minimal + defensible):
  * Learn the segment rules on the SEASONED population only: months_on_book >= 3,
    i.e. loans that have had a genuine window to reveal a 90+ default.
    (MOB>=3 keeps 82% of the book and all three products - BNPL 68%, Personal/SME
    ~89%; MOB>=6 or >=12 would erase BNPL, which is short-lived.)
  * ASSIGN every loan (seasoned or not) to the resulting segments, so F/G/I still
    have full coverage.
  * REPORT each segment's default rate on its seasoned loans only (the honest
    basis), alongside n_seasoned so the basis is transparent. Value metrics use
    the frozen value_proxy on all loans (value is not censored the way 90+ is).

Unchanged design rules: origination-knowable features only (no behaviour, no
outcomes, no z); value_proxy frozen and used only to characterise.

Run from the wsE root:   python methods/train_tree_segments.py
Reads:   wsE/out/frames/loan_frame.parquet
Writes:  wsE/out/methods/  (assignments, metrics, rules, value-drivers, manifest)
"""

from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text

# ======================================================================
# CONFIG
# ======================================================================
SEED = 20260603
ROOT   = Path(__file__).resolve().parent.parent
FRAMES = ROOT / "out" / "frames"
OUTDIR = ROOT / "out" / "methods"
CRORE  = 10_000_000

SEASONED_MOB_MIN = 3          # 90+ DPD is unobservable below this; data shows 0.00% at MOB 0-2
MIN_SEGMENTS, MAX_SEGMENTS = 4, 6
TREE_PARAMS = dict(max_leaf_nodes=MAX_SEGMENTS, min_samples_leaf=1000,
                   max_depth=4, class_weight="balanced", random_state=SEED)

NUMERIC_FEATURES  = ["ticket_size", "tenure_mo", "interest_rate_apr",
                     "approval_tat_hrs", "age", "first_time_borrower", "loan_sequence"]
ORDINAL_GRADES    = ["origination_risk_grade", "credit_quality_indicator"]
ONEHOT_FEATURES   = ["product_type", "region_type", "income_proxy_band",
                     "employment_type", "acquisition_channel"]
GRADE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
GRADE_INV = {v: k for k, v in GRADE_MAP.items()}
FORBIDDEN = {"true_latent_risk", "z", "value_proxy", "default_flag", "loan_status",
             "worst_delq_bucket", "worst_delq_rank", "ever_90plus", "months_on_book",
             "nii", "fee", "loss", "servicing", "cac_charged", "margin_before_cac"}


def _fail(msg):
    print(f"\n  [X] CHECK FAILED: {msg}"); sys.exit(1)


# ======================================================================
# feature matrix
# ======================================================================
def build_design_matrix(lf: pd.DataFrame):
    cols, names = [], []
    for c in NUMERIC_FEATURES:
        cols.append(lf[c].astype(float).values.reshape(-1, 1)); names.append(c)
    for c in ORDINAL_GRADES:
        cols.append(lf[c].astype(str).map(GRADE_MAP).values.reshape(-1, 1)); names.append(c + "_ord")
    for c in ONEHOT_FEATURES:
        d = pd.get_dummies(lf[c].astype(str), prefix=c, prefix_sep="=")
        cols.append(d.values); names.extend(list(d.columns))
    X = np.hstack(cols).astype(float)
    leaked = [n for n in names if n.split("=")[0].lower() in FORBIDDEN or n.lower() in FORBIDDEN]
    if leaked: _fail(f"forbidden feature leaked into the tree: {leaked}")
    return X, names


# ======================================================================
# readable rules
# ======================================================================
def leaf_paths(tree, feature_names):
    t = tree.tree_; out = {}
    def rec(node, conds):
        if t.children_left[node] == t.children_right[node]:
            out[node] = list(conds); return
        f, thr = feature_names[t.feature[node]], t.threshold[node]
        rec(t.children_left[node],  conds + [(f, "<=", thr)])
        rec(t.children_right[node], conds + [(f, ">",  thr)])
    rec(0, []); return out

def humanise(conds):
    onehot, numeric = {}, {}
    for f, op, thr in conds:
        if "=" in f:
            col, val = f.split("=", 1); side = "is" if op == ">" else "not"
            onehot.setdefault(col, {}).setdefault(side, set()).add(val)
        else:
            lo, hi = numeric.get(f, (-np.inf, np.inf))
            if op == "<=": hi = min(hi, thr)
            else:          lo = max(lo, thr)
            numeric[f] = (lo, hi)
    parts = []
    for col, d in onehot.items():
        if "is" in d:  parts.append(f"{col} is {', '.join(sorted(d['is']))}")
        if "not" in d: parts.append(f"{col} not {', '.join(sorted(d['not']))}")
    for c in ("origination_risk_grade_ord", "credit_quality_indicator_ord"):
        if c in numeric:
            lo, hi = numeric[c]; base = c.replace("_ord", "")
            los = GRADE_INV.get(int(np.ceil(lo + 1e-9)) if lo > -np.inf else 1, "A")
            his = GRADE_INV.get(int(np.floor(hi + 1e-9)) if hi < np.inf else 5, "E")
            parts.append(f"{base} {los}-{his}" if los != his else f"{base} {los}")
    fmt = lambda f, v: (f"\u20b9{v:,.0f}" if f == "ticket_size" else
                        f"{v:.1f}%" if f == "interest_rate_apr" else f"{v:,.0f}")
    for f, (lo, hi) in numeric.items():
        if f.endswith("_ord"): continue
        if lo > -np.inf and hi < np.inf: parts.append(f"{f} {fmt(f, lo)}-{fmt(f, hi)}")
        elif hi < np.inf:                parts.append(f"{f} \u2264 {fmt(f, hi)}")
        elif lo > -np.inf:               parts.append(f"{f} > {fmt(f, lo)}")
    return "; ".join(parts) if parts else "(all loans)"


# ======================================================================
# metrics: default on SEASONED loans; value on ALL loans
# ======================================================================
def segment_metrics(lf, seg_col):
    total_n, total_orig = len(lf), lf.ticket_size.sum()
    rows = []
    for seg, d in lf.groupby(seg_col, sort=True):
        s = d[d.seasoned == 1]
        rows.append(dict(
            segment=seg, n_loans=len(d), n_seasoned=len(s), n_customers=d.customer_id.nunique(),
            pct_of_book=round(100 * len(d) / total_n, 1),
            total_originated_cr=round(d.ticket_size.sum() / CRORE, 1),
            pct_of_orig=round(100 * d.ticket_size.sum() / total_orig, 1),
            default_rate_pct=round(100 * s.default_flag.mean(), 2) if len(s) else np.nan,
            ever_90_rate_pct=round(100 * s.ever_90plus.mean(), 2) if len(s) else np.nan,
            mean_ticket_inr=round(d.ticket_size.mean()),
            mean_apr_pct=round(d.interest_rate_apr.mean(), 1),
            mean_value_proxy_inr=round(d.value_proxy.mean()),
            total_value_proxy_cr=round(d.value_proxy.sum() / CRORE, 2),
            pct_loss_making=round(100 * (d.value_proxy < 0).mean(), 1),
        ))
    return pd.DataFrame(rows)


# ======================================================================
# main
# ======================================================================
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    lf = pd.read_parquet(FRAMES / "loan_frame.parquet")
    if [c for c in lf.columns if c.lower() in ("true_latent_risk", "z")]:
        _fail("z present in loan_frame")
    lf["seasoned"] = (lf.months_on_book >= SEASONED_MOB_MIN).astype(int)
    seas = lf[lf.seasoned == 1]
    print(f"Loaded {len(lf):,} loans | seasoned (MOB>={SEASONED_MOB_MIN}): {len(seas):,} "
          f"({len(seas)/len(lf)*100:.0f}%) used to learn the rules")

    X_all, names = build_design_matrix(lf)
    X_seas = X_all[(lf.seasoned == 1).values]
    y_seas = seas.default_flag.astype(int).values

    # ---- risk tree learned on SEASONED loans ----
    clf = DecisionTreeClassifier(**TREE_PARAMS).fit(X_seas, y_seas)
    leaf_all = clf.apply(X_all)                       # assign EVERY loan
    n_leaves = len(np.unique(leaf_all))
    if not (MIN_SEGMENTS <= n_leaves <= MAX_SEGMENTS):
        _fail(f"tree produced {n_leaves} leaves, outside {MIN_SEGMENTS}-{MAX_SEGMENTS}")

    paths = leaf_paths(clf, names)
    seasoned_def = {}
    for lid in np.unique(leaf_all):
        mask = (leaf_all == lid) & (lf.seasoned.values == 1)
        seasoned_def[lid] = lf.default_flag.values[mask].mean() if mask.any() else np.nan
    order = sorted(np.unique(leaf_all), key=lambda l: seasoned_def[l])
    rules = {lid: humanise(paths[lid]) for lid in order}
    label = {lid: f"T{i+1}: {rules[lid]}" for i, lid in enumerate(order)}
    lf["segment"] = pd.Series(leaf_all, index=lf.index).map(label)

    metrics = segment_metrics(lf, "segment")

    # ---- value drivers (seasoned, for Step 7) ----
    reg = DecisionTreeRegressor(max_leaf_nodes=MAX_SEGMENTS, min_samples_leaf=1000,
                                max_depth=4, random_state=SEED).fit(X_seas, seas.value_proxy.values)
    vimp = sorted(zip(names, reg.feature_importances_), key=lambda x: -x[1])[:6]

    # ---- cross-checks ----
    print("\nCross-checks")
    if lf.segment.isna().any(): _fail("some loans unassigned")
    if int(metrics.n_loans.sum()) != len(lf): _fail("assignments don't cover all loans once")
    print(f"  [ok] {n_leaves} segments | every loan assigned once ({len(lf):,})")
    if (metrics.default_rate_pct <= 0).any():
        _fail("a segment still shows 0% seasoned default -> artifact not fixed")
    print(f"  [ok] artifact gone: every segment seasoned-default in "
          f"{metrics.default_rate_pct.min()}%-{metrics.default_rate_pct.max()}% "
          f"(spread {metrics.default_rate_pct.max()-metrics.default_rate_pct.min():.1f}pp)")
    if (metrics.n_seasoned < 200).any():
        print("  [warn] a segment has <200 seasoned loans - rate is thin, note at selection")
    tot_val = round(lf.value_proxy.sum() / CRORE, 2)
    if abs(metrics.total_value_proxy_cr.sum() - tot_val) > 0.05: _fail("value does not reconcile")
    print(f"  [ok] value reconciles: {metrics.total_value_proxy_cr.sum():.2f} cr (~\u20b9473.6M)")
    print("  [ok] rules learned only on observable-outcome loans; no behaviour/outcome/z input")

    # ---- write ----
    lf[["loan_id", "customer_id", "segment"]].to_csv(OUTDIR / "tree_segment_assignments.csv", index=False)
    metrics.to_csv(OUTDIR / "tree_segment_metrics.csv", index=False)
    with open(OUTDIR / "tree_rules.txt", "w") as f:
        f.write(f"WS-E Step 4 (v2, seasoned MOB>={SEASONED_MOB_MIN}) - Tree risk segments\n")
        f.write("Default rates are on seasoned loans only; segments cover all loans.\n")
        f.write("=" * 66 + "\n\n")
        for i, lid in enumerate(order):
            r = metrics.iloc[i]
            f.write(f"T{i+1}  ({int(r.n_loans):,} loans, {r.pct_of_book}% of book | "
                    f"{int(r.n_seasoned):,} seasoned)\n")
            f.write(f"     RULE : {rules[lid]}\n")
            f.write(f"     RISK : default {r.default_rate_pct}% (seasoned) | ever-90+ {r.ever_90_rate_pct}%\n")
            f.write(f"     VALUE: mean \u20b9{int(r.mean_value_proxy_inr):,} | {r.pct_loss_making}% loss-making\n\n")
        f.write("\nRaw tree (sklearn export_text):\n" + export_text(clf, feature_names=names, max_depth=4))
    with open(OUTDIR / "tree_value_drivers.txt", "w") as f:
        f.write("Top features that split VALUE (seasoned regression tree, for Step 7):\n\n")
        for nm, imp in vimp: f.write(f"  {imp*100:5.1f}%  {nm}\n")
    json.dump({"step": "WS-E Step 4 v2 - tree segmentation (seasoned)", "seed": SEED,
               "seasoning_rule": f"months_on_book >= {SEASONED_MOB_MIN}",
               "seasoned_basis_share": round(len(seas) / len(lf), 3),
               "n_segments": int(n_leaves), "features_used": names,
               "target": "default_flag (ever 90+ DPD), learned on seasoned loans",
               "default_rate_basis": "seasoned loans only", "value_basis": "frozen value_proxy, all loans",
               "z_excluded": True, "behaviour_excluded": True},
              open(OUTDIR / "tree_manifest.json", "w"), indent=2)

    pd.set_option("display.width", 230); pd.set_option("display.max_columns", None)
    print("\nTree segments (v2, seasoned; provisional T-labels):")
    print(metrics[["segment", "n_loans", "n_seasoned", "pct_of_book", "default_rate_pct",
                   "mean_value_proxy_inr", "pct_loss_making"]].to_string(index=False))
    print("\nValue drivers (top):", ", ".join(f"{n} {i*100:.0f}%" for n, i in vimp[:4]))
    print(f"\nWrote -> {OUTDIR}\nStep 4 (v2) complete. Artifact fixed. All checks passed.")


if __name__ == "__main__":
    main()
