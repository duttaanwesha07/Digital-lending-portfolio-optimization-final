"""
generate/assemble.py — Step 6 of the build.

Turns the working tables into the delivered dataset:

  1. Derived fields are already in place (origination_cohort, months_on_book);
     roll_rate is analysis-time, not stored.
  2. Drop the internal z (true_latent_risk) from the delivered customers table;
     stash it to out/meta/_true_latent_risk.parquet so the Gate-1 harness
     (Step 8) can still verify the z-driven relationships (A-019, §11.1).
  3. Trim each table to its spec §5 columns (drop internal helpers like
     period_month_idx / month_idx, and the loan-grain default_flag that the
     engine carried on repayment rows). The value_proxy components are moved to
     an audit file rather than the delivered loans table.
  4. Apply §11.3–§11.4 conventions: whole-rupee currency, 0/1 int flags,
     ISO-8601 dates in CSV, YYYY-MM months.
  5. Export every table as Parquet (authoritative dtypes) + CSV.
  6. Write data_dictionary (CSV + JSON) from §5 and run_manifest.json.

Idempotent: if z has already been dropped, the stash step is skipped.
"""

from pathlib import Path
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

ROOT = Path(__file__).resolve().parent.parent
PQ = ROOT / "out" / "data" / "parquet"
CSV = ROOT / "out" / "data" / "csv"
META = ROOT / "out" / "meta"

# Delivered column sets, in spec §5 order ------------------------------------
COLS_CUSTOMERS = ["customer_id", "region_type", "income_proxy_band",
                  "employment_type", "age", "first_time_borrower",
                  "credit_quality_indicator", "acquisition_channel", "cac"]
COLS_LOANS = ["loan_id", "customer_id", "loan_sequence", "product_type",
              "ticket_size", "tenure_mo", "interest_rate_apr",
              "origination_risk_grade", "origination_date", "origination_cohort",
              "approval_tat_hrs", "months_on_book", "loan_status",
              "default_flag", "value_proxy"]
COLS_REPAYMENTS = ["loan_id", "period_index", "period_date", "scheduled_emi",
                   "outstanding_balance", "amount_paid", "dpd",
                   "delinquency_bucket", "paid_on_time_flag",
                   "partial_payment_flag", "missed_payments_cum",
                   "prepayment_flag"]
COLS_BEHAVIOUR = ["customer_id", "month", "cashflow_consistency",
                  "balance_volatility", "spending_shock_flag",
                  "nach_bounce_count", "credit_utilisation",
                  "bureau_inquiry_velocity", "app_engagement"]

RUPEE_INT = {  # currency fields rounded to whole rupees (§11.4)
    "customers": ["cac"],
    "loans": ["ticket_size", "value_proxy"],
    "repayments": ["scheduled_emi", "outstanding_balance", "amount_paid"],
}
FLAG_INT = {
    "customers": ["first_time_borrower"],
    "loans": ["default_flag"],
    "repayments": ["paid_on_time_flag", "partial_payment_flag", "prepayment_flag"],
    "behaviour_monthly": ["spending_shock_flag"],
}


def _apply_conventions(df, table):
    for c in RUPEE_INT.get(table, []):
        if c in df:
            df[c] = np.round(df[c]).astype("int64")
    for c in FLAG_INT.get(table, []):
        if c in df:
            df[c] = df[c].astype("int8")
    return df


def assemble():
    META.mkdir(parents=True, exist_ok=True)
    CSV.mkdir(parents=True, exist_ok=True)

    customers = pd.read_parquet(PQ / "customers.parquet")
    loans = pd.read_parquet(PQ / "loans.parquet")
    repayments = pd.read_parquet(PQ / "repayments.parquet")
    behaviour = pd.read_parquet(PQ / "behaviour_monthly.parquet")

    # --- 1. stash + drop internal z ----------------------------------------
    if "true_latent_risk" in customers.columns:
        customers[["customer_id", "true_latent_risk"]].to_parquet(
            META / "_true_latent_risk.parquet", index=False)
        print("  stashed internal z -> meta/_true_latent_risk.parquet, dropping from delivery")
    # --- 1b. move value_proxy components to an audit file ------------------
    audit_cols = [c for c in ["nii", "fee", "loss", "servicing", "cac_charged"] if c in loans.columns]
    if audit_cols:
        loans[["loan_id"] + audit_cols].to_csv(META / "value_proxy_components.csv", index=False)
        print(f"  wrote value_proxy audit components -> meta/value_proxy_components.csv")

    # --- 2. trim to delivered columns --------------------------------------
    customers = customers[COLS_CUSTOMERS].copy()
    loans = loans[COLS_LOANS].copy()
    repayments = repayments[COLS_REPAYMENTS].copy()
    behaviour = behaviour[COLS_BEHAVIOUR].copy()

    # --- 3. conventions -----------------------------------------------------
    customers = _apply_conventions(customers, "customers")
    loans = _apply_conventions(loans, "loans")
    repayments = _apply_conventions(repayments, "repayments")
    behaviour = _apply_conventions(behaviour, "behaviour_monthly")

    tables = {"customers": customers, "loans": loans,
              "repayments": repayments, "behaviour_monthly": behaviour}

    # --- 4/5. export parquet (typed) + csv (ISO dates) ---------------------
    for name, df in tables.items():
        df.to_parquet(PQ / f"{name}.parquet", index=False)
        csv_df = df.copy()
        for c in csv_df.columns:
            if pd.api.types.is_datetime64_any_dtype(csv_df[c]):
                csv_df[c] = pd.to_datetime(csv_df[c]).dt.strftime("%Y-%m-%d")
        csv_df.to_csv(CSV / f"{name}.csv", index=False)
        print(f"  exported {name}: {len(df):,} rows  (parquet + csv)")

    # --- 6. data dictionary + manifest -------------------------------------
    _write_data_dictionary()
    _write_manifest(tables)
    return tables


def _write_data_dictionary():
    """Machine-readable dictionary derived from spec §5."""
    rows = [
        # customers
        ("customers", "customer_id", "string", "C+6 digits, unique", "M", "Structural", "Customer key"),
        ("customers", "region_type", "categorical", "Urban/Semi-urban/Rural", "M", "Independent root", "Geography"),
        ("customers", "income_proxy_band", "ordinal", "Low/Mid/High", "M", "Independent root", "Income proxy"),
        ("customers", "employment_type", "categorical", "Salaried/Self-employed/Gig/Informal", "M", "Independent root", "Employment"),
        ("customers", "age", "integer", "18-65", "N", "Independent root", "Age in years"),
        ("customers", "first_time_borrower", "flag", "0/1", "M", "Independent root", "New-to-credit"),
        ("customers", "credit_quality_indicator", "ordinal", "A-E", "M", "Risk-engine", "Bureau band (noisy obs of z)"),
        ("customers", "acquisition_channel", "categorical", "Digital ads/Partner-embedded/Referral/DSA/Organic", "M", "Independent root", "Acquisition channel"),
        ("customers", "cac", "int (INR)", ">=0", "M", "Derived", "Cost of acquisition"),
        # loans
        ("loans", "loan_id", "string", "L+7 digits, unique", "M", "Structural", "Loan key"),
        ("loans", "customer_id", "string", "FK->customers", "M", "Structural", "Owning customer"),
        ("loans", "loan_sequence", "integer", "1-3", "M", "Derived", "1=first loan"),
        ("loans", "product_type", "categorical", "Personal/BNPL/SME", "M", "Independent root", "Product"),
        ("loans", "ticket_size", "int (INR)", "product bands", "M", "Derived", "Loan amount"),
        ("loans", "tenure_mo", "integer", "product bands", "M", "Derived", "Contractual months"),
        ("loans", "interest_rate_apr", "float (%)", "12-48", "M", "Risk-engine", "Priced APR"),
        ("loans", "origination_risk_grade", "ordinal", "A-E", "M", "Risk-engine", "Grade at underwriting"),
        ("loans", "origination_date", "date", "ISO-8601", "M", "Independent root", "Origination day"),
        ("loans", "origination_cohort", "string", "YYYY-MM", "M", "Derived", "Vintage month"),
        ("loans", "approval_tat_hrs", "float (hrs)", "~0.1-72", "M", "Derived", "Onboarding TAT"),
        ("loans", "months_on_book", "integer", ">=0", "M", "Derived", "MOB at snapshot/closure"),
        ("loans", "loan_status", "categorical", "Active/Closed/Written-off", "M", "Derived", "As-of-snapshot status"),
        ("loans", "default_flag", "flag", "0/1", "M", "Risk-engine", "Ever 90+ DPD (sticky)"),
        ("loans", "value_proxy", "int (INR)", "may be negative", "M", "Derived", "Risk-adj. contribution"),
        # repayments
        ("repayments", "loan_id", "string", "FK->loans", "M", "Structural", "Owning loan"),
        ("repayments", "period_index", "integer", "1..tenure", "M", "Structural", "Scheduled period"),
        ("repayments", "period_date", "date", "ISO-8601", "M", "Derived", "Period date"),
        ("repayments", "scheduled_emi", "int (INR)", ">0", "M", "Derived", "Amortised EMI"),
        ("repayments", "outstanding_balance", "int (INR)", ">=0", "M", "Derived", "Opening principal; EAD basis"),
        ("repayments", "amount_paid", "int (INR)", ">=0", "M", "Risk-engine", "Paid this period"),
        ("repayments", "dpd", "integer", ">=0", "M", "Risk-engine", "Days past due"),
        ("repayments", "delinquency_bucket", "ordinal", "Current/1-30/31-60/61-90/90+", "M", "Derived", "DPD bucket"),
        ("repayments", "paid_on_time_flag", "flag", "0/1", "M", "Derived", "Full & current"),
        ("repayments", "partial_payment_flag", "flag", "0/1", "M", "Derived", "Partial payment"),
        ("repayments", "missed_payments_cum", "integer", ">=0", "M", "Derived", "Cumulative misses"),
        ("repayments", "prepayment_flag", "flag", "0/1", "N", "Risk-engine", "Early settlement"),
        # behaviour_monthly
        ("behaviour_monthly", "customer_id", "string", "FK->customers", "M", "Structural", "Owning customer"),
        ("behaviour_monthly", "month", "string", "YYYY-MM", "M", "Structural", "Snapshot month"),
        ("behaviour_monthly", "cashflow_consistency", "float", "0-1", "M", "Risk-engine+shock", "Inflow regularity; falls pre-default"),
        ("behaviour_monthly", "balance_volatility", "float", ">=0", "M", "Risk-engine+shock", "Balance std; rises pre-default"),
        ("behaviour_monthly", "spending_shock_flag", "flag", "0/1", "M", "Risk-engine+shock", "Outflow spike"),
        ("behaviour_monthly", "nach_bounce_count", "integer", "0-5", "M", "Risk-engine+shock", "Auto-debit failures (3m)"),
        ("behaviour_monthly", "credit_utilisation", "float (%)", "0-150", "N", "Risk-engine+shock", "Revolving utilisation"),
        ("behaviour_monthly", "bureau_inquiry_velocity", "integer", "0-10", "N", "Risk-engine+shock", "Inquiries last 90d"),
        ("behaviour_monthly", "app_engagement", "float", ">=0", "N", "Risk-engine+shock", "Sessions/month"),
    ]
    dd = pd.DataFrame(rows, columns=["table", "variable", "type", "allowed_values",
                                     "must_nice", "derivation", "description"])
    dd.to_csv(META / "data_dictionary.csv", index=False)
    with open(META / "data_dictionary.json", "w") as f:
        json.dump(dd.to_dict(orient="records"), f, indent=2)
    print(f"  wrote data_dictionary.csv / .json ({len(dd)} fields)")


def _write_manifest(tables):
    manifest = {
        "dataset": "Digital Lending Portfolio Optimization — synthetic",
        "master_seed": config.MASTER_SEED,
        "spec_version": "1.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generation_order": ["customers", "loans", "engine(repayments+behaviour)",
                             "outcomes", "assemble"],
        "row_counts": {name: int(len(df)) for name, df in tables.items()},
        "internal_fields_excluded": ["true_latent_risk (stashed in meta/_true_latent_risk.parquet)"],
        "gate1_result": None,  # filled by Step 8
        "files": {
            "parquet": [f"out/data/parquet/{n}.parquet" for n in tables],
            "csv": [f"out/data/csv/{n}.csv" for n in tables],
            "meta": ["out/meta/data_dictionary.csv", "out/meta/data_dictionary.json",
                     "out/meta/run_manifest.json", "out/meta/value_proxy_components.csv",
                     "out/meta/_true_latent_risk.parquet"],
        },
    }
    with open(META / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  wrote run_manifest.json (seed {config.MASTER_SEED}, spec v1.1)")


def verify():
    """Quick post-export verification of the delivered set."""
    print("\nVERIFY delivered dataset:")
    cust = pd.read_parquet(PQ / "customers.parquet")
    print("  z absent from delivered customers:", "true_latent_risk" not in cust.columns)
    for name, cols in [("customers", COLS_CUSTOMERS), ("loans", COLS_LOANS),
                       ("repayments", COLS_REPAYMENTS), ("behaviour_monthly", COLS_BEHAVIOUR)]:
        p = pd.read_parquet(PQ / f"{name}.parquet")
        c = pd.read_csv(CSV / f"{name}.csv", nrows=5)
        ok = list(p.columns) == cols
        print(f"  {name:18} cols match §5: {ok}   parquet+csv present: {(PQ/f'{name}.parquet').exists() and (CSV/f'{name}.csv').exists()}")


if __name__ == "__main__":
    print("Assembling delivered dataset...")
    tables = assemble()
    verify()
    print("\nStep 6 complete — dataset, dictionary and manifest written.")
