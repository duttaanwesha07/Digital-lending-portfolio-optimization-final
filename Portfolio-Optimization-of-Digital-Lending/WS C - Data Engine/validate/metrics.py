"""
validate/metrics.py — reusable measurements used by the Gate-1 harness.

Pure read-only analysis of the delivered dataset (+ the internal z stashed in
out/meta). No randomness except a fixed-seed sample for the lead-time check.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PQ = ROOT / "out" / "data" / "parquet"
META = ROOT / "out" / "meta"


def load_tables():
    t = {
        "customers": pd.read_parquet(PQ / "customers.parquet"),
        "loans": pd.read_parquet(PQ / "loans.parquet"),
        "repayments": pd.read_parquet(PQ / "repayments.parquet"),
        "behaviour": pd.read_parquet(PQ / "behaviour_monthly.parquet"),
    }
    z = pd.read_parquet(META / "_true_latent_risk.parquet")
    t["z"] = z.set_index("customer_id")["true_latent_risk"]
    return t


def auc(scores, labels):
    """ROC-AUC via Mann–Whitney U (no sklearn). Higher score → more likely 1."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    pos, neg = labels == 1, labels == 0
    n_pos, n_neg = pos.sum(), neg.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty_like(order, float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def month_str(dt):
    return pd.to_datetime(dt).dt.strftime("%Y-%m")


def loan_behaviour_feature(t):
    """Per-loan single-feature risk score from the customer's behaviour over the
    loan's active window: uses (1 - min cashflow_consistency) and max balance_volatility.
    Returns DataFrame[loan_id, score, default_flag]."""
    rep = t["repayments"][["loan_id", "period_date"]].copy()
    rep["m"] = month_str(rep["period_date"])
    rep = rep.merge(t["loans"][["loan_id", "customer_id", "default_flag"]], on="loan_id")
    beh = t["behaviour"][["customer_id", "month", "cashflow_consistency", "balance_volatility"]]
    j = rep.merge(beh, left_on=["customer_id", "m"], right_on=["customer_id", "month"], how="left")
    g = j.groupby("loan_id").agg(
        min_cf=("cashflow_consistency", "min"),
        max_bv=("balance_volatility", "max"),
        default_flag=("default_flag", "first"),
    ).dropna()
    g["score"] = (1 - g["min_cf"]) + g["max_bv"]
    return g.reset_index()


def lead_time_days(t, sample=4000, seed=7):
    """Median days between first behavioural deterioration and the 90+ event,
    among defaulted loans."""
    rep = t["repayments"][["loan_id", "period_index", "period_date", "dpd"]].copy()
    rep["m"] = month_str(rep["period_date"])
    rep = rep.merge(t["loans"][["loan_id", "customer_id"]], on="loan_id")
    first90 = (rep[rep["dpd"] >= 90].sort_values(["loan_id", "period_index"])
               .groupby("loan_id").first()[["customer_id", "m"]])
    if first90.empty:
        return float("nan"), 0, 0
    beh = t["behaviour"].copy()
    beh["mi"] = pd.to_datetime(beh["month"] + "-01")
    bi = beh.set_index(["customer_id", "month"])["cashflow_consistency"]
    s = first90.sample(min(sample, len(first90)), random_state=seed)
    leads = []
    for lid, row in s.iterrows():
        cid = row["customer_id"]
        m90 = pd.to_datetime(row["m"] + "-01")
        onset = None
        for back in range(1, 7):
            mm = (m90 - pd.DateOffset(months=back)).strftime("%Y-%m")
            if (cid, mm) in bi.index and float(bi.loc[(cid, mm)]) < 0.60:
                onset = back
        if onset is not None:
            leads.append(onset * 30)
    leads = np.array(leads)
    return (float(np.median(leads)) if len(leads) else float("nan"),
            len(leads), len(s))


def cure_fraction_proxy(t):
    """Share of loans that touched delinquency (dpd>0) but never reached 90+.
    Proxy for the false-positive realism (design ~35% at episode level)."""
    mx = t["repayments"].groupby("loan_id")["dpd"].max()
    delinq = mx[mx > 0]
    return float((delinq < 90).mean()) if len(delinq) else float("nan")
