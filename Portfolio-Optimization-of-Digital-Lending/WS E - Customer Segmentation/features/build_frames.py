#!/usr/bin/env python3
"""
Workstream E - Segmentation | Step 2: Build the analytical frames
=================================================================
Data Lead track (Part 1).

Turns the four frozen Workstream-C tables into THREE clean analytical frames:

  1. customer_frame        - one row per customer  (value / CLV segmentation)
  2. loan_frame            - one row per loan       (risk segmentation)
  3. product_cell_frame    - one row per product x tenure-band x ticket-band (Q3)

Design rules enforced (from the Part 1 manual):
  * value_proxy is read FROZEN from the data; it is NEVER recomputed.
  * true_latent_risk (z) is NEVER read or attached - validation-only (A-019).
  * CAC is counted once per customer (it is charged only on the first loan),
    so CLV = sum(value_proxy) double-counts nothing (A-058 / A-015).
  * outcome fields (default_flag, value_proxy, loan_status, worst delinquency)
    are kept as CHARACTERISATION columns, never as segment-defining features.
  * product/tenure/ticket bands reuse Workstream D's exact definitions so the
    product cells reconcile with WS D Slices A/B/C.

Run from the wsE root:   python features/build_frames.py
Inputs  read from:       wsE/inputs/
Outputs written to:      wsE/out/frames/
"""

from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

# ======================================================================
# CONFIG  (these mirror your Step-1 config.py decisions; kept here so the
#          script runs standalone. Move to config.py later if you prefer.)
# ======================================================================
SEED = 20260603                       # master seed (A-062) - recorded, not re-rolled here

ROOT    = Path(__file__).resolve().parent.parent        # .../wsE
INPUTS  = ROOT / "inputs"
OUTDIR  = ROOT / "out" / "frames"

# Expected row counts from run_manifest.json (integrity gate)
EXPECTED_ROWS = {"customers": 40000, "loans": 49600,
                 "repayments": 381032, "behaviour_monthly": 350187}

# --- column roles (documented so the Risk Consultant can co-sign the split) ---
# Borrower / loan attributes knowable at or near origination -> may DEFINE segments.
CUSTOMER_PROFILE_FEATURES = [
    "region_type", "income_proxy_band", "employment_type", "age",
    "first_time_borrower", "credit_quality_indicator", "acquisition_channel",
]
LOAN_TERM_FEATURES = [
    "product_type", "ticket_size", "tenure_mo", "interest_rate_apr",
    "origination_risk_grade", "approval_tat_hrs", "loan_sequence",
]
# Stable behavioural summaries - usable as descriptors; the *dynamic* early-warning
# signal is Workstream F, not here.
BEHAVIOUR_SUMMARY_FEATURES = [
    "cashflow_consistency_mean", "balance_volatility_mean",
    "credit_utilisation_mean", "app_engagement_mean",
    "bureau_inquiry_velocity_mean", "nach_bounce_total",
    "spending_shock_rate", "n_months_observed",
]
# Realised outcomes - CHARACTERISE segments, never define them.
CHARACTERISATION_COLS = [
    "default_flag", "loan_status", "value_proxy", "months_on_book",
    "worst_delq_bucket", "ever_90plus",
]
# Hard exclusion - never a feature, never read.
FORBIDDEN_FEATURES = ["true_latent_risk", "z", "_true_latent_risk"]

# --- Workstream D bucket conventions (verbatim, so cells reconcile) ----
PRODUCTS = ["BNPL", "Personal", "SME"]
MIN_BAND_SHARE = 0.05
TICKET_BANDS = {
    "BNPL":     ([10_000, 25_000],     ["<10k", "10-25k", ">25k"]),
    "Personal": ([50_000, 150_000],    ["<50k", "50-150k", ">150k"]),
    "SME":      ([250_000, 1_000_000], ["<250k", "250k-1M", ">1M"]),
}
TENURE_ORDER = {"BNPL": ["1-3", "4-6"], "Personal": ["3-12", "13-24", "25-36"],
                "SME": ["6-12", "13-24", "25-36+"]}
DELQ_ORDER = ["Current", "1-30", "31-60", "61-90", "90+"]
DELQ_RANK  = {b: i for i, b in enumerate(DELQ_ORDER)}
SEASONED_MOB = 12                     # WS D equal-MOB anchor for default rates


# ======================================================================
# helpers
# ======================================================================
def tenure_bucket(p, t):
    if p == "BNPL":     return "1-3" if t <= 3 else "4-6"
    if p == "Personal": return "3-12" if t <= 12 else ("13-24" if t <= 24 else "25-36")
    if p == "SME":      return "6-12" if t <= 12 else ("13-24" if t <= 24 else "25-36+")
    return None

def ticket_band_for(series: pd.Series, product: str) -> pd.Series:
    """Documented bands, with WS D's auto-fallback to data tertiles if degenerate."""
    edges, labels = TICKET_BANDS[product]
    bk = pd.cut(series, [-np.inf] + edges + [np.inf], labels=labels)
    if (bk.value_counts(normalize=True) < MIN_BAND_SHARE).any():
        q = series.quantile([1/3, 2/3]).round(0).tolist()
        labels = [f"<{int(q[0]):,}", f"{int(q[0]):,}-{int(q[1]):,}", f">{int(q[1]):,}"]
        bk = pd.cut(series, [-np.inf] + q + [np.inf], labels=labels)
        print(f"  [guard] {product}: documented ticket band degenerate -> data thirds {labels}")
    return bk.astype(str)

def _fail(msg):
    print(f"\n  [X] CHECK FAILED: {msg}")
    sys.exit(1)


# ======================================================================
# load + integrity gate
# ======================================================================
def load_inputs():
    print("Loading inputs from", INPUTS)
    need = ["customers.parquet", "loans.parquet", "repayments.parquet",
            "behaviour_monthly.parquet", "value_proxy_components.csv"]
    missing = [f for f in need if not (INPUTS / f).exists()]
    if missing:
        _fail(f"missing input files in {INPUTS}: {missing}")

    cust = pd.read_parquet(INPUTS / "customers.parquet")
    loans = pd.read_parquet(INPUTS / "loans.parquet")
    rep = pd.read_parquet(INPUTS / "repayments.parquet")
    beh = pd.read_parquet(INPUTS / "behaviour_monthly.parquet")
    vc = pd.read_csv(INPUTS / "value_proxy_components.csv")

    # --- integrity gate: counts, keys, no z leakage ---
    for name, df in [("customers", cust), ("loans", loans),
                     ("repayments", rep), ("behaviour_monthly", beh)]:
        if len(df) != EXPECTED_ROWS[name]:
            _fail(f"{name} row count {len(df):,} != manifest {EXPECTED_ROWS[name]:,}")
    if cust.customer_id.duplicated().any():  _fail("duplicate customer_id")
    if loans.loan_id.duplicated().any():     _fail("duplicate loan_id")
    if vc.loan_id.duplicated().any():        _fail("duplicate loan_id in value components")
    if loans.customer_id.isin(cust.customer_id).all() is np.False_:
        _fail("orphan loans (customer_id not in customers)")
    for df in (cust, loans, rep, beh, vc):
        bad = [c for c in df.columns if c.lower() in FORBIDDEN_FEATURES]
        if bad: _fail(f"forbidden field present in an input: {bad}")

    # --- value reconciliation: frozen value_proxy vs components (<=1 rupee, WS D rule) ---
    chk = loans[["loan_id", "value_proxy"]].merge(vc, on="loan_id", validate="one_to_one")
    recomputed = chk.nii + chk.fee - chk.loss - chk.servicing - chk.cac_charged
    gap = (chk.value_proxy - recomputed).abs().max()
    if gap > 1.0:
        _fail(f"value_proxy does not reconcile to components (max gap {gap:.3f})")
    print(f"  integrity OK | value reconciles within {gap:.3f} rupee | no z leakage")

    # --- CAC charged once: must be zero on every non-first loan (A-058) ---
    seq_cac = loans.merge(vc[["loan_id", "cac_charged"]], on="loan_id")
    if seq_cac.loc[seq_cac.loan_sequence > 1, "cac_charged"].abs().sum() != 0:
        _fail("cac_charged is non-zero on a non-first loan -> CLV would double-count CAC")
    print("  CAC attribution OK | charged only on first loan -> CLV counts it once")
    return cust, loans, rep, beh, vc


# ======================================================================
# 1. loan-grain frame  (risk segmentation)
# ======================================================================
def build_loan_frame(cust, loans, rep, vc, beh_cust):
    # worst delinquency ever, per loan, from repayments
    r = rep[["loan_id", "delinquency_bucket"]].copy()
    r["rank"] = r.delinquency_bucket.map(DELQ_RANK)
    worst = r.groupby("loan_id")["rank"].max()
    inv = {v: k for k, v in DELQ_RANK.items()}

    lf = loans.copy()
    lf["product_type"] = lf.product_type.astype(str)
    lf["worst_delq_rank"] = lf.loan_id.map(worst).fillna(0).astype(int)
    lf["worst_delq_bucket"] = lf.worst_delq_rank.map(inv)
    lf["ever_90plus"] = (lf.worst_delq_rank >= DELQ_RANK["90+"]).astype("int8")
    lf["seasoned_mob12"] = (lf.months_on_book >= SEASONED_MOB).astype("int8")

    # value components (frozen value_proxy already on loans; add the breakdown)
    lf = lf.merge(vc, on="loan_id", how="left", validate="one_to_one")
    lf["margin_before_cac"] = lf.nii + lf.fee - lf.loss - lf.servicing

    # product-specific bands (match WS D)
    lf["tenure_band"] = [tenure_bucket(p, t) for p, t in zip(lf.product_type, lf.tenure_mo)]
    kb = pd.Series(index=lf.index, dtype=object)
    for p in PRODUCTS:
        idx = lf.product_type == p
        kb[idx] = ticket_band_for(lf.loc[idx, "ticket_size"], p).values
    lf["ticket_band"] = kb

    # denormalise borrower profile + behaviour summary onto the loan
    lf = lf.merge(cust[["customer_id"] + CUSTOMER_PROFILE_FEATURES],
                  on="customer_id", how="left", validate="many_to_one")
    lf = lf.merge(beh_cust, on="customer_id", how="left", validate="many_to_one")

    if lf.loan_id.duplicated().any(): _fail("loan_frame: duplicate loan_id")
    if len(lf) != EXPECTED_ROWS["loans"]: _fail("loan_frame: row count drift")
    return lf


# ======================================================================
# behaviour summary  (stable per-customer descriptors)
# ======================================================================
def build_behaviour_summary(beh):
    g = beh.groupby("customer_id")
    bs = pd.DataFrame({
        "cashflow_consistency_mean": g.cashflow_consistency.mean(),
        "balance_volatility_mean":   g.balance_volatility.mean(),
        "credit_utilisation_mean":   g.credit_utilisation.mean(),
        "app_engagement_mean":       g.app_engagement.mean(),
        "bureau_inquiry_velocity_mean": g.bureau_inquiry_velocity.mean(),
        "nach_bounce_total":         g.nach_bounce_count.sum(),
        "spending_shock_rate":       g.spending_shock_flag.mean(),
        "n_months_observed":         g.size(),
    }).reset_index()
    return bs


# ======================================================================
# 2. customer-grain frame  (value / CLV segmentation)
# ======================================================================
def build_customer_frame(cust, loan_frame, beh_cust):
    lf = loan_frame
    g = lf.groupby("customer_id")
    rel = pd.DataFrame({
        # relationship shape
        "n_loans":            g.loan_id.size(),
        "n_products_held":    g.product_type.nunique(),
        "primary_product":    g.apply(lambda d: d.sort_values("loan_sequence").product_type.iloc[0]),
        "total_originated":   g.ticket_size.sum(),
        "mean_apr":           g.interest_rate_apr.mean(),
        "first_cohort":       g.origination_cohort.min(),
        "max_months_on_book": g.months_on_book.max(),
        # value (frozen, CAC counted once) - A-058
        "clv":          g.value_proxy.sum(),
        "total_nii":    g.nii.sum(),
        "total_fee":    g.fee.sum(),
        "total_loss":   g.loss.sum(),
        "total_servicing": g.servicing.sum(),
        "total_cac":    g.cac_charged.sum(),
        "margin_before_cac_total": g.margin_before_cac.sum(),
        # risk characterisation
        "any_default":  g.default_flag.max(),
        "n_defaults":   g.default_flag.sum(),
        "n_writeoffs":  g.loan_status.apply(lambda s: (s == "Written-off").sum()),
        "worst_delq_rank_ever": g.worst_delq_rank.max(),
        "ever_90plus":  g.ever_90plus.max(),
    }).reset_index()
    inv = {v: k for k, v in DELQ_RANK.items()}
    rel["worst_delq_bucket_ever"] = rel.worst_delq_rank_ever.map(inv)
    rel["clv_to_cac"] = np.where(rel.total_cac > 0, rel.clv / rel.total_cac, np.nan)
    rel["net_value_positive"] = (rel.clv > 0).astype("int8")

    cf = cust.merge(rel, on="customer_id", how="left", validate="one_to_one")
    cf = cf.merge(beh_cust, on="customer_id", how="left", validate="one_to_one")
    cf["has_behaviour"] = cf.n_months_observed.notna().astype("int8")

    if cf.customer_id.duplicated().any(): _fail("customer_frame: duplicate customer_id")
    if len(cf) != EXPECTED_ROWS["customers"]: _fail("customer_frame: row count drift")
    return cf


# ======================================================================
# 3. product-cell frame  (product x tenure-band x ticket-band -> Q3)
# ======================================================================
def build_product_cell_frame(loan_frame):
    lf = loan_frame
    total = len(lf)
    g = lf.groupby(["product_type", "tenure_band", "ticket_band"], observed=True)
    cell = pd.DataFrame({
        "n_loans":        g.loan_id.size(),
        "default_rate":   g.default_flag.mean(),
        "ever_90_rate":   g.ever_90plus.mean(),
        "mean_value_proxy": g.value_proxy.mean(),
        "total_value_proxy": g.value_proxy.sum(),
        "mean_ticket":    g.ticket_size.mean(),
        "mean_tenure":    g.tenure_mo.mean(),
        "mean_apr":       g.interest_rate_apr.mean(),
        "mean_mob":       g.months_on_book.mean(),
        "n_seasoned":     g.seasoned_mob12.sum(),
    })
    # seasoned default rate (MOB>=12) per cell - avoids immature-cohort bias
    seas = lf[lf.seasoned_mob12 == 1].groupby(
        ["product_type", "tenure_band", "ticket_band"], observed=True).default_flag.mean()
    cell["default_rate_seasoned"] = seas
    cell = cell.reset_index()
    cell["share_of_book"] = cell.n_loans / total
    cell["loss_making"] = (cell.mean_value_proxy < 0).astype("int8")

    if cell.n_loans.sum() != total:
        _fail("product cells do not cover every loan exactly once")
    return cell.sort_values(["product_type", "tenure_band", "ticket_band"]).reset_index(drop=True)


# ======================================================================
# cross-checks + write
# ======================================================================
def cross_checks(loan_frame, customer_frame, cell_frame, loans):
    print("\nCross-checks")
    # conservation: customer CLV total == loan value_proxy total
    a, b = customer_frame.clv.sum(), loans.value_proxy.sum()
    if int(a) != int(b): _fail(f"CLV conservation broke: {a:,} vs {b:,}")
    print(f"  [ok] CLV conservation: sum(CLV)=sum(value_proxy)={int(b):,}")
    # no forbidden field anywhere
    for nm, df in [("loan", loan_frame), ("customer", customer_frame), ("cell", cell_frame)]:
        bad = [c for c in df.columns if c.lower() in FORBIDDEN_FEATURES]
        if bad: _fail(f"forbidden field in {nm}_frame: {bad}")
    print("  [ok] no true_latent_risk / z column in any frame")
    # every loan has both bands
    if loan_frame[["tenure_band", "ticket_band"]].isna().any().any():
        _fail("some loans have a missing band")
    print("  [ok] every loan assigned a tenure_band and ticket_band")
    # grains
    print(f"  [ok] grains: {len(customer_frame):,} customers | "
          f"{len(loan_frame):,} loans | {len(cell_frame)} product cells")


def write_manifest(loan_frame, customer_frame, cell_frame):
    def roles(df, profile, terms):
        out = {}
        for c in df.columns:
            if c in ("customer_id", "loan_id"): out[c] = "KEY"
            elif c in profile or c in terms: out[c] = "FEATURE"
            elif c in BEHAVIOUR_SUMMARY_FEATURES: out[c] = "BEHAVIOUR_SUMMARY"
            elif c in CHARACTERISATION_COLS or c.lower() in (
                 "value_proxy", "default_flag", "loan_status", "clv", "any_default",
                 "n_defaults", "n_writeoffs", "worst_delq_bucket_ever", "ever_90plus",
                 "worst_delq_rank", "worst_delq_rank_ever", "worst_delq_bucket"):
                out[c] = "CHARACTERISATION"
            else: out[c] = "DERIVED"
        return out
    manifest = {
        "step": "WS-E Step 2 - build_frames",
        "seed": SEED,
        "z_excluded": True,
        "value_proxy": "frozen from source; never recomputed",
        "cac": "charged once per customer (first loan) -> CLV counts it once",
        "frames": {
            "customer_frame": {"rows": int(len(customer_frame)),
                               "column_roles": roles(customer_frame, CUSTOMER_PROFILE_FEATURES, [])},
            "loan_frame": {"rows": int(len(loan_frame)),
                           "column_roles": roles(loan_frame, CUSTOMER_PROFILE_FEATURES, LOAN_TERM_FEATURES)},
            "product_cell_frame": {"rows": int(len(cell_frame))},
        },
    }
    (OUTDIR / "frames_manifest.json").write_text(json.dumps(manifest, indent=2))


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cust, loans, rep, beh, vc = load_inputs()

    beh_cust = build_behaviour_summary(beh)
    loan_frame = build_loan_frame(cust, loans, rep, vc, beh_cust)
    customer_frame = build_customer_frame(cust, loan_frame, beh_cust)
    cell_frame = build_product_cell_frame(loan_frame)

    cross_checks(loan_frame, customer_frame, cell_frame, loans)

    loan_frame.to_parquet(OUTDIR / "loan_frame.parquet", index=False)
    customer_frame.to_parquet(OUTDIR / "customer_frame.parquet", index=False)
    cell_frame.to_parquet(OUTDIR / "product_cell_frame.parquet", index=False)
    cell_frame.to_csv(OUTDIR / "product_cell_frame.csv", index=False)  # human-readable
    write_manifest(loan_frame, customer_frame, cell_frame)

    print("\nWrote ->", OUTDIR)
    for f in ["customer_frame.parquet", "loan_frame.parquet",
              "product_cell_frame.parquet", "product_cell_frame.csv", "frames_manifest.json"]:
        print("   ", f)
    print("\nStep 2 complete. All checks passed.")


if __name__ == "__main__":
    main()
