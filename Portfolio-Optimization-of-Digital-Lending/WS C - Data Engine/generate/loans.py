"""
generate/loans.py — Step 3 of the build.

Builds the ~50,000-row loans table:

  1. Each customer gets 1 / 2 / 3 loans per the 80 / 16 / 4 split.
  2. For each loan:
       - product_type from PRODUCT_MIX
       - origination_risk_grade from (z + underwriting noise), banded
       - ticket_size and tenure_mo from product bands, with grade-based shrink
       - interest_rate_apr = APR base + grade-step + small noise
       - origination_date in one of 24 monthly vintages, with mild growth
       - approval_tat_hrs lognormal within the customer's channel TAT band
  3. Loans within a customer are dated in increasing order; loan_sequence
     reflects that chronological order. Repeat loans pay no CAC (handled
     later in Step 5 when value_proxy is computed).
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    REPEAT_LOAN_SPLIT,
    PRODUCT_MIX, TICKET, TENURE, GRADE_TERM_SHRINK,
    GRADES, GRADE_MIX, GRADE_CUTPOINTS_Z, GRADE_UNDERWRITING_NOISE_SD,
    APR_BASE, APR_GRADE_STEP, APR_NOISE_SD, APR_MIN, APR_MAX,
    CHANNEL_TAT_BAND,
    N_VINTAGES, VINTAGE_GROWTH,
    get_rng,
)


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------

def _bucket_into_grades(z_obs):
    cp = GRADE_CUTPOINTS_Z
    out = np.empty(len(z_obs), dtype="U1")
    out[z_obs >= cp["A"]] = "A"
    out[(z_obs >= cp["B"]) & (z_obs < cp["A"])] = "B"
    out[(z_obs >= cp["C"]) & (z_obs < cp["B"])] = "C"
    out[(z_obs >= cp["D"]) & (z_obs < cp["C"])] = "D"
    out[z_obs < cp["D"]] = "E"
    return out


# ----------------------------------------------------------------------------
# per-attribute samplers
# ----------------------------------------------------------------------------

def _sample_loan_counts(n_customers, rng):
    keys = np.array(list(REPEAT_LOAN_SPLIT.keys()))
    probs = np.array(list(REPEAT_LOAN_SPLIT.values()))
    return rng.choice(keys, size=n_customers, p=probs)


def _sample_products(n, rng):
    return rng.choice(list(PRODUCT_MIX.keys()), size=n, p=list(PRODUCT_MIX.values()))


def _sample_ticket(products, grades, rng):
    """Lognormal around the product median × grade_factor, clipped."""
    n = len(products)
    out = np.empty(n, dtype=np.int64)
    for prod, params in TICKET.items():
        mask = products == prod
        if not mask.any():
            continue
        factors = np.array([GRADE_TERM_SHRINK[g]["ticket_factor"] for g in grades[mask]])
        med_adj = params["median"] * factors
        draws = rng.lognormal(mean=np.log(med_adj), sigma=params["sigma"])
        draws = np.clip(draws, params["min"], params["max"])
        out[mask] = np.round(draws).astype(np.int64)
    return out


def _sample_tenure(products, grades, rng):
    """Triangular within range, peaked at mode, shrunk for worse grades."""
    n = len(products)
    out = np.empty(n, dtype=np.int32)
    for prod, params in TENURE.items():
        mask = products == prod
        if not mask.any():
            continue
        lo, hi, mode = params["min"], params["max"], params["mode"]
        max_factors = np.array([GRADE_TERM_SHRINK[g]["tenure_max_factor"] for g in grades[mask]])
        hi_adj = np.maximum(lo + 1, np.round(hi * max_factors).astype(int))
        mode_adj = np.clip(mode, lo + 1, hi_adj - 1).astype(int)
        draws = rng.triangular(left=lo, mode=mode_adj, right=hi_adj)
        out[mask] = np.clip(np.round(draws), lo, hi_adj).astype(np.int32)
    return out


def _sample_apr(products, grades, rng):
    grade_idx = np.array([GRADES.index(g) for g in grades])
    base = np.array([APR_BASE[p] for p in products])
    step = np.array([APR_GRADE_STEP[p] for p in products])
    noise = rng.normal(0.0, APR_NOISE_SD, len(products))
    apr = base + grade_idx * step + noise
    return np.clip(apr, APR_MIN, APR_MAX).round(2)


def _sample_tat(channels, rng):
    """Lognormal within the channel TAT band; band ≈ 10th–90th percentile."""
    out = np.empty(len(channels), dtype=np.float64)
    for ch, (lo, hi) in CHANNEL_TAT_BAND.items():
        mask = channels == ch
        if not mask.any():
            continue
        med = np.sqrt(lo * hi)
        sigma = np.log(hi / med) / 1.28
        draws = rng.lognormal(mean=np.log(med), sigma=sigma, size=mask.sum())
        out[mask] = np.clip(draws, lo * 0.5, hi * 2.0)
    return out.round(2)


def _sample_vintages_per_customer(loan_counts, rng):
    """Per customer, draw k vintages with growth-weighted probabilities, sort them.

    Vintage k weight ∝ (1 + VINTAGE_GROWTH/24)^k, so newer cohorts are larger.
    Returns: list[np.ndarray] of length n_customers; each array sorted ascending.
    """
    weights = np.array([(1 + VINTAGE_GROWTH / N_VINTAGES) ** k for k in range(N_VINTAGES)])
    weights /= weights.sum()
    vintages_flat = rng.choice(np.arange(1, N_VINTAGES + 1), size=int(loan_counts.sum()), p=weights)
    out = []
    idx = 0
    for k in loan_counts:
        v = np.sort(vintages_flat[idx:idx + k])
        out.append(v)
        idx += k
    return out


def _vintages_to_dates(per_customer_vintages, rng):
    """Convert per-customer sorted vintage arrays into concrete dates.

    Snapshot reference: end of vintage 24 = 2026-06-30.
    Within a vintage month, sample day-of-month uniformly. For multi-loan
    customers, day offsets within the SAME vintage are sampled in sorted order
    so chronological order is preserved (loan_sequence will be re-assigned by
    sorting after).
    """
    snapshot = pd.Timestamp("2026-06-30")
    vintage_start_month = (snapshot.replace(day=1)
                           - pd.DateOffset(months=N_VINTAGES - 1))

    all_dates = []
    for vintages in per_customer_vintages:
        if len(vintages) == 0:
            continue
        # for each loan, get its month
        months = [vintage_start_month + pd.DateOffset(months=int(v - 1)) for v in vintages]
        # sample day-of-month for each loan, sorted within ties so seq order = date order
        day_draws = rng.integers(0, [m.days_in_month for m in months])
        # within ties (same vintage), sort the day draws
        # group consecutive vintages
        sorted_days = day_draws.copy()
        i = 0
        while i < len(vintages):
            j = i
            while j + 1 < len(vintages) and vintages[j + 1] == vintages[i]:
                j += 1
            if j > i:
                sorted_days[i:j + 1] = np.sort(day_draws[i:j + 1])
            i = j + 1
        for m, d in zip(months, sorted_days):
            all_dates.append(m + pd.Timedelta(days=int(d)))
    return pd.DatetimeIndex(all_dates)


# ----------------------------------------------------------------------------
# main builder
# ----------------------------------------------------------------------------

def make_loans(customers_df):
    n_customers = len(customers_df)

    # how many loans per customer
    loan_counts = _sample_loan_counts(n_customers, get_rng("loans", "counts"))
    total_loans = int(loan_counts.sum())

    # expand customers to per-loan
    cust_idx = np.repeat(np.arange(n_customers), loan_counts)
    customer_id = customers_df["customer_id"].to_numpy()[cust_idx]
    z = customers_df["true_latent_risk"].to_numpy()[cust_idx]
    channel = customers_df["acquisition_channel"].to_numpy()[cust_idx]

    # product, grade, ticket, tenure, apr
    product = _sample_products(total_loans, get_rng("loans", "product"))
    underwriting_noise = get_rng("loans", "grade_noise").normal(
        0.0, GRADE_UNDERWRITING_NOISE_SD, total_loans
    )
    grade = _bucket_into_grades(z + underwriting_noise)
    ticket = _sample_ticket(product, grade, get_rng("loans", "ticket"))
    tenure = _sample_tenure(product, grade, get_rng("loans", "tenure"))
    apr = _sample_apr(product, grade, get_rng("loans", "apr"))
    tat = _sample_tat(channel, get_rng("loans", "tat"))

    # origination dates with mild growth, sorted within each customer
    per_cust_vintages = _sample_vintages_per_customer(loan_counts, get_rng("loans", "vintages"))
    orig_dates = _vintages_to_dates(per_cust_vintages, get_rng("loans", "dates"))

    df = pd.DataFrame({
        "loan_id":                 [f"L{i:07d}" for i in range(1, total_loans + 1)],
        "customer_id":             customer_id,
        "loan_sequence":           np.zeros(total_loans, dtype=np.int8),  # filled below
        "product_type":            product,
        "ticket_size":             ticket,
        "tenure_mo":               tenure,
        "origination_risk_grade":  pd.Categorical(grade, categories=GRADES, ordered=True),
        "interest_rate_apr":       apr.astype(np.float32),
        "origination_date":        orig_dates,
        "origination_cohort":      orig_dates.strftime("%Y-%m"),
        "approval_tat_hrs":        tat.astype(np.float32),
    })

    # Sort by (customer_id, origination_date) then renumber loan_sequence so
    # seq order matches chronological order even when two loans share a month.
    df = df.sort_values(["customer_id", "origination_date", "loan_id"]).reset_index(drop=True)
    df["loan_sequence"] = (df.groupby("customer_id").cumcount() + 1).astype(np.int8)

    return df


# ----------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------

def _fmt_mix(series, target_dict, tol=0.03):
    actual = series.value_counts(normalize=True).sort_index()
    lines = []
    for label, target in target_dict.items():
        a = float(actual.get(label, 0.0))
        flag = "✓" if abs(a - target) <= tol else "✗"
        lines.append(f"    {label:<14} {a:6.1%}   (target {target:.0%}, ±{tol:.0%}) {flag}")
    return "\n".join(lines)


def report(df, customers_df):
    n = len(df)
    n_customers_with_loan = df["customer_id"].nunique()
    print(f"Loans built: {n:,}")
    print(f"Loans per customer: {n / len(customers_df):.3f}   (target ~1.24)")
    print(f"Customers holding ≥1 loan: {n_customers_with_loan:,}\n")

    print("Loan-count distribution per customer:")
    counts = df.groupby("customer_id").size().value_counts(normalize=True).sort_index()
    for k in [1, 2, 3]:
        target = REPEAT_LOAN_SPLIT[k]
        actual = float(counts.get(k, 0.0))
        flag = "✓" if abs(actual - target) <= 0.03 else "✗"
        print(f"    {k} loan(s): {actual:6.1%}   (target {target:.0%}, ±3pp) {flag}")

    print("\nProduct mix:")
    print(_fmt_mix(df["product_type"], PRODUCT_MIX))

    print("\nOrigination-risk-grade mix:")
    print(_fmt_mix(df["origination_risk_grade"], GRADE_MIX))

    print("\nAPR by product × grade (median):")
    for prod in APR_BASE:
        line = [f"    {prod:<10}"]
        for g in GRADES:
            mask = (df["product_type"] == prod) & (df["origination_risk_grade"] == g)
            if mask.sum() == 0:
                line.append(f"{g}=  -  ")
                continue
            med = df.loc[mask, "interest_rate_apr"].median()
            line.append(f"{g}={med:5.1f}%")
        print("   ".join(line))
    print("    (expected base + step × index: BNPL 24/27/30/33/36, "
          "Personal 16/20/24/28/32, SME 14/17/20/23/26)")

    print("\nTicket median by product × grade (should drop for D/E):")
    for prod in TICKET:
        line = [f"    {prod:<10}"]
        for g in GRADES:
            mask = (df["product_type"] == prod) & (df["origination_risk_grade"] == g)
            if mask.sum() == 0:
                line.append(f"{g}=     -    ")
                continue
            med = int(df.loc[mask, "ticket_size"].median())
            line.append(f"{g}=₹{med:>8,}")
        print("   ".join(line))

    print("\nTenure median by product (months):")
    for prod in TENURE:
        sub = df[df["product_type"] == prod]["tenure_mo"]
        print(f"    {prod:<10} median {int(sub.median())}  range [{int(sub.min())}, {int(sub.max())}]   "
              f"(mode target {TENURE[prod]['mode']}, range [{TENURE[prod]['min']}, {TENURE[prod]['max']}])")

    print("\nVintage spread (loans per month):")
    vc = df["origination_cohort"].value_counts().sort_index()
    print(f"    earliest cohort {vc.index[0]}: {vc.iloc[0]:,} loans")
    print(f"    latest cohort   {vc.index[-1]}: {vc.iloc[-1]:,} loans")
    print(f"    ratio newest/oldest: {vc.iloc[-1]/vc.iloc[0]:.2f}   (target ~{1 + VINTAGE_GROWTH:.2f})")

    # TAT by channel: join with customers
    df_with_ch = df.merge(customers_df[["customer_id", "acquisition_channel"]], on="customer_id")
    print("\nApproval TAT by channel (median hours):")
    for ch, (lo, hi) in CHANNEL_TAT_BAND.items():
        sub = df_with_ch[df_with_ch["acquisition_channel"] == ch]["approval_tat_hrs"]
        if len(sub):
            print(f"    {ch:<18} median {sub.median():5.2f} h  range [{sub.min():.1f}, {sub.max():.1f}]   (band [{lo}, {hi}])")

    # Ordering check
    multi = df[df.duplicated(subset="customer_id", keep=False)].sort_values(["customer_id", "loan_sequence"])
    out_of_order = 0
    for _, grp in multi.groupby("customer_id"):
        dates = grp["origination_date"].to_numpy()
        if any(dates[i] > dates[i + 1] for i in range(len(dates) - 1)):
            out_of_order += 1
    print(f"\nLoan-sequence ordering check: {out_of_order} customers with out-of-order dates "
          f"(must be 0)   {'✓' if out_of_order == 0 else '✗'}")

    print("\nFirst 8 rows:")
    cols = ["loan_id", "customer_id", "loan_sequence", "product_type",
            "ticket_size", "tenure_mo", "origination_risk_grade",
            "interest_rate_apr", "origination_date", "approval_tat_hrs"]
    print(df[cols].head(8).to_string(index=False))


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    customers_df = pd.read_parquet(root / "out" / "data" / "parquet" / "customers.parquet")
    print(f"Loaded {len(customers_df):,} customers\n")

    df = make_loans(customers_df)
    report(df, customers_df)

    out_path = root / "out" / "data" / "parquet" / "loans.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nWritten: {out_path}")
