"""
generate/outcomes.py — Step 5 of the build.

Finalises the loan-level outcomes and the value measure, then writes them onto
the loans table (spec §5.2 fields: default_flag, loan_status, value_proxy,
months_on_book).

value_proxy is computed in ONE function (A-009 / §9.3) so the number is
identical everywhere it is used downstream (one number, one source):

    value_proxy = NII + Fee - Loss - Servicing - CAC

    NII       = Σ over panel periods of (APR - CoF)/12 × outstanding_balance_t
    Fee       = fee_rate(product) × ticket_size          (once, at origination)
    Loss      = Written-off  -> LGD × outstanding_balance @ write-off
                currently 90+ -> LGD × last outstanding_balance
                else          -> 0
    Servicing = ₹80 × active loan-months
    CAC       = cac if loan_sequence == 1 else 0          (A-016)

default_flag and loan_status were produced by the engine (Step 4); here we read
them back, confirm the rules, and attach them. default_flag is sticky and is
NOT reversed by cure (A-056); loan_status precedence is Written-off > Closed >
Active (A-057), independent of default_flag.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    LGD_BY_PRODUCT, FEE_RATE_BY_PRODUCT, SERVICING_PER_ACTIVE_MONTH,
    COST_OF_FUNDS_ANNUAL, DEFAULT_DPD_THRESHOLD,
)


def compute_outcomes(loans_df, customers_df, repayments_df, loan_status, loan_default):
    """Return loans_df enriched with default_flag, loan_status, value_proxy,
    months_on_book. All value_proxy components computed once, here."""

    # --- per-loan aggregates from the repayment panel -----------------------
    rep = repayments_df.merge(
        loans_df[["loan_id", "product_type", "interest_rate_apr"]],
        on="loan_id", how="left",
    )
    cof_monthly = COST_OF_FUNDS_ANNUAL / 12.0
    # monthly net interest = (APR - CoF)/12 × outstanding, accrued while on book
    rep["_nii"] = (rep["interest_rate_apr"] / 100.0 - COST_OF_FUNDS_ANNUAL) / 12.0 \
        * rep["outstanding_balance"]

    agg = rep.groupby("loan_id").agg(
        nii=("_nii", "sum"),
        active_months=("period_index", "size"),
    )
    # last observed row per loan (max period_index) → exposure & current dpd
    last = (rep.sort_values(["loan_id", "period_index"])
               .groupby("loan_id").tail(1)
               .set_index("loan_id")[["outstanding_balance", "dpd"]]
               .rename(columns={"outstanding_balance": "last_outstanding",
                                "dpd": "last_dpd"}))

    df = loans_df.set_index("loan_id").join(agg).join(last)

    # cac lives at customer grain; charged to the first loan only (A-016)
    cac_by_cust = customers_df.set_index("customer_id")["cac"]
    df["_cac_full"] = df["customer_id"].map(cac_by_cust).fillna(0.0)
    df["cac_charged"] = np.where(df["loan_sequence"] == 1, df["_cac_full"], 0.0)

    # status & default from the engine
    df["loan_status"] = df.index.map(loan_status)
    df["default_flag"] = df.index.map(loan_default).astype(int)

    # --- value_proxy components ---------------------------------------------
    df["nii"] = df["nii"].fillna(0.0)
    df["active_months"] = df["active_months"].fillna(0).astype(int)
    df["fee"] = df["product_type"].map(FEE_RATE_BY_PRODUCT) * df["ticket_size"]
    df["servicing"] = SERVICING_PER_ACTIVE_MONTH * df["active_months"]

    lgd = df["product_type"].map(LGD_BY_PRODUCT)
    is_writeoff = df["loan_status"] == "Written-off"
    is_current_90 = (~is_writeoff) & (df["last_dpd"] >= DEFAULT_DPD_THRESHOLD)
    df["loss"] = np.where(is_writeoff | is_current_90,
                          lgd * df["last_outstanding"].fillna(0.0), 0.0)

    df["value_proxy"] = (df["nii"] + df["fee"] - df["loss"]
                         - df["servicing"] - df["cac_charged"]).round(2)

    # months_on_book = periods observed (to termination / snapshot)
    df["months_on_book"] = df["active_months"]

    # tidy: keep the spec §5.2 loan columns + the value components for audit
    keep = list(loans_df.columns) + [
        "loan_status", "default_flag", "value_proxy", "months_on_book",
        "nii", "fee", "loss", "servicing", "cac_charged",
    ]
    keep = [c for c in keep if c != "loan_id"]
    out = df[keep].reset_index()  # loan_id back as column
    return out


def report(out):
    print(f"Loans with outcomes: {len(out):,}")
    print("\nVP1 — value_proxy nulls:", int(out["value_proxy"].isna().sum()), "(must be 0)")
    print("VP1 — reconciles to component sum:",
          bool(np.allclose(
              out["value_proxy"],
              (out["nii"] + out["fee"] - out["loss"] - out["servicing"] - out["cac_charged"]).round(2),
              atol=0.5)))

    pos = (out["value_proxy"] > 0).mean()
    neg = (out["value_proxy"] < 0).mean()
    print(f"\nVP2 — sign distribution: {pos:.1%} positive / {neg:.1%} negative (must be two-signed)")
    wo = out[out["loan_status"] == "Written-off"]["value_proxy"]
    print(f"      written-off mean value_proxy: ₹{wo.mean():,.0f}  ({(wo<0).mean():.0%} negative)")

    print("\nVP3 — mean value_proxy by grade (A should beat E):")
    g = out.groupby("origination_risk_grade", observed=True)["value_proxy"].mean()
    for gr in ["A", "B", "C", "D", "E"]:
        if gr in g.index:
            print(f"    {gr}  ₹{g[gr]:>12,.0f}")
    print(f"      A > E: {bool(g.get('A', 0) > g.get('E', 0))}")

    print("\nMean value_proxy by product:")
    for p, gg in out.groupby("product_type", observed=True):
        print(f"    {p:<10} ₹{gg['value_proxy'].mean():>12,.0f}   "
              f"(default {gg['default_flag'].mean():.1%}, mean MOB {gg['months_on_book'].mean():.1f})")

    print("\nComponent means (₹): "
          f"NII {out['nii'].mean():,.0f} | Fee {out['fee'].mean():,.0f} | "
          f"Loss {out['loss'].mean():,.0f} | Servicing {out['servicing'].mean():,.0f} | "
          f"CAC {out['cac_charged'].mean():,.0f}")

    # default_flag vs loan_status are independent dimensions (§9.2)
    print("\ndefault_flag × loan_status cross-tab (counts):")
    print(pd.crosstab(out["default_flag"], out["loan_status"]).to_string())


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    pq = root / "out" / "data" / "parquet"
    loans_df = pd.read_parquet(pq / "loans.parquet")
    customers_df = pd.read_parquet(pq / "customers.parquet")
    repayments_df = pd.read_parquet(pq / "repayments.parquet")
    loan_status = pd.read_parquet(pq / "_loan_status.parquet")["loan_status"].to_dict()
    loan_default = pd.read_parquet(pq / "_loan_default.parquet")["default_flag"].to_dict()
    print(f"Loaded {len(loans_df):,} loans, {len(repayments_df):,} repayment rows\n")

    out = compute_outcomes(loans_df, customers_df, repayments_df, loan_status, loan_default)
    report(out)

    out.to_parquet(pq / "loans.parquet", index=False)
    print(f"\nWritten: loans.parquet (now carries loan_status, default_flag, "
          f"value_proxy, months_on_book + value components)")
