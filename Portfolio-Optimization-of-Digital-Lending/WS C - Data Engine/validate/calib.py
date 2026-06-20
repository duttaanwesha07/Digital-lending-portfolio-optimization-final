"""
validate/calib.py — fast Step-7 calibration loop (dev tool, not a build step).

Runs the real generators + engine on a ~5k-customer sample so each tuning pass
takes seconds, and prints the metrics that map to the failing Gate-1 checks
(DR1/DR2/DR3, R6/R12 AUC, R5 lead, S7 panel-volume ratio). Tune config, re-run
this, repeat; confirm on the full pipeline + full harness at the end.

Usage:  python validate/calib.py [n_customers]
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "generate"))
import importlib
import config; importlib.reload(config)
import customers as C, loans as L, engine as E

GRADES = ["A", "B", "C", "D", "E"]


def auc(scores, labels):
    scores = np.asarray(scores, float); labels = np.asarray(labels, int)
    pos, neg = labels == 1, labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = scores.argsort(); ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    return (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum())


def run(n_cust=5000):
    cust = C.make_customers().head(n_cust).copy()
    loans = L.make_loans(cust)
    rep, beh, status, deflt = E.run_engine(cust, loans)

    loans = loans.set_index("loan_id")
    loans["default_flag"] = pd.Series(deflt)
    loans["loan_status"] = pd.Series(status)
    loans = loans.reset_index()

    blended = loans.default_flag.mean()
    bp = loans.groupby("product_type", observed=True)["default_flag"].mean()
    bg = loans.groupby("origination_risk_grade", observed=True)["default_flag"].mean()

    # single-feature AUC: per-customer min cashflow + max volatility vs any-default
    cd = loans.groupby("customer_id")["default_flag"].max()
    bagg = beh.groupby("customer_id").agg(min_cf=("cashflow_consistency", "min"),
                                          max_bv=("balance_volatility", "max"))
    bagg["score"] = (1 - bagg["min_cf"]) + bagg["max_bv"]
    bagg["d"] = cd.reindex(bagg.index).fillna(0).astype(int)
    a = auc(bagg.score, bagg.d)

    # lead time
    rep2 = rep
    f90 = rep2[rep2.dpd >= 90].groupby("loan_id").first()[["customer_id", "period_month_idx"]]
    bi = beh.set_index(["customer_id", "month_idx"])["cashflow_consistency"]
    leads = []
    for _, r in f90.iterrows():
        c, m = r.customer_id, int(r.period_month_idx); onset = None
        for b in range(1, 7):
            if (c, m - b) in bi.index and float(bi.loc[(c, m - b)]) < 0.60:
                onset = b
        if onset: leads.append(onset * 30)
    lead = np.median(leads) if leads else float("nan")

    print(f"\n--- sample: {n_cust:,} customers, {len(loans):,} loans ---")
    print(f"DR1 blended : {blended:5.2%}   (target 7.5%, band 6-9)")
    print(f"DR2 product : BNPL {bp.get('BNPL',0):5.2%} | Personal {bp.get('Personal',0):5.2%} | SME {bp.get('SME',0):5.2%}")
    print(f"            target BNPL~10 (highest) > Personal~7 > SME~5; ordered={bp.get('BNPL',0)>bp.get('Personal',0)>bp.get('SME',0)}")
    print(f"DR3 grade   : " + " ".join(f"{g}={bg.get(g,0):5.2%}" for g in GRADES))
    tgt = config.DEFAULT_TARGET_BY_GRADE
    rel = {g: f"{(bg.get(g,0)-tgt[g])/tgt[g]:+.0%}" for g in GRADES}
    print(f"            rel-to-target: {rel}  (need within ±20%, increasing)")
    print(f"R6/R12 AUC  : {a:5.3f}   (need 0.60 < AUC < 0.95)")
    print(f"R5 lead     : {lead:.0f} days   (need >=60)")
    print(f"S7 ratios   : repay/loan={len(rep)/len(loans):.1f} (full→{len(rep)/len(loans)*49600/1000:.0f}k vs 550k) | "
          f"beh/cust={len(beh)/n_cust:.1f} (full→{len(beh)/n_cust*40000/1000:.0f}k vs 480k)")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    run(n)
