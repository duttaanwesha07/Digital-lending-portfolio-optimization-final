import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# === Where your loan_frame lives — update if it's not here ===
LOAN_FRAME = os.path.join(HERE, "out", "data", "parquet", "loan_frame.parquet")
# If it's in wsE instead, use:
# LOAN_FRAME = os.path.join(HERE, "..", "wsE", "out", "frames", "loan_frame.parquet")

# === 1. Load loan frame ===
lf = pd.read_parquet(LOAN_FRAME, engine="fastparquet")
print(f"Total loans: {len(lf):,}")

# === 2. Apply hybrid segmentation rules ===
# Backbone = product × grade (your Step 3 baseline)
# Added split = high-APR pocket (>32.4%), which overrides everything else
APR_CUTOFF = 32.4

def hybrid_segment(row):
    if row["interest_rate_apr"] > APR_CUTOFF:
        return "7. High-APR Danger (>32.4%)"
    p, g = row["product_type"], row["origination_risk_grade"]
    if p == "BNPL":
        return "1. BNPL (APR ≤ 32.4%)"
    if p == "Personal":
        if g in ("A","B"): return "2. Personal Prime (A-B)"
        if g == "C":       return "3. Personal Mid (C)"
        if g in ("D","E"): return "4. Personal Subprime (D-E)"
    if p == "SME":
        if g in ("A","B"): return "5. SME Prime (A-B)"
        if g in ("C","D","E"): return "6. SME Non-prime (C-D-E)"
    return "UNASSIGNED"

lf["segment"] = lf.apply(hybrid_segment, axis=1)
assert (lf["segment"] != "UNASSIGNED").all(), "Some loans were not assigned"

# === 3. Compute metrics on seasoned basis (MOB >= 3) ===
total_per_segment = lf.groupby("segment").size().to_dict()
seasoned = lf[lf["months_on_book"] >= 3]

metrics = seasoned.groupby("segment").agg(
    n_loans_seasoned=("loan_id", "count"),
    default_rate_pct=("default_flag", lambda x: round(100 * x.mean(), 2)),
    mean_value_proxy_inr=("value_proxy", lambda x: round(x.mean(), 0)),
    pct_loss_making=("value_proxy", lambda x: round(100 * (x < 0).mean(), 1)),
).reset_index()

metrics["n_loans_total"] = metrics["segment"].map(total_per_segment)
metrics["pct_of_book"] = (100 * metrics["n_loans_total"] / len(lf)).round(1)
metrics = metrics[["segment", "n_loans_total", "pct_of_book",
                   "default_rate_pct", "mean_value_proxy_inr", "pct_loss_making"]]

print("\n=== Hybrid segmentation — metrics ===")
print(metrics.to_string(index=False))

# === 4. Save outputs ===
metrics.to_csv(os.path.join(HERE, "step6_hybrid_metrics.csv"), index=False)
lf[["loan_id", "customer_id", "segment"]].to_csv(
    os.path.join(HERE, "step6_hybrid_assignments.csv"), index=False
)
print(f"\nSaved: step6_hybrid_metrics.csv, step6_hybrid_assignments.csv")