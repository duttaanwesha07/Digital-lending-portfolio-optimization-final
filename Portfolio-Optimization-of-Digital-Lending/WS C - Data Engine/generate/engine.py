"""
generate/engine.py — Step 4 of the build (THE CORE).

Simulates every loan month by month and emits the two panels TOGETHER, in one
loop per customer, so a customer's behaviour row and their loan's repayment row
line up on the same calendar month (no look-ahead, two-hop EWS join valid).

The realism core (spec §6.5–§6.6, A-024/A-025):

  - Each customer has a latent stress process s(t) >= 0, ~0 in normal months.
  - With a probability that rises as z falls, the customer enters a stress
    episode: s ramps up over 2–4 months. THIS RAMP IS THE LEAD WINDOW — it
    happens BEFORE the missed payments that cascade to 90+, so the behavioural
    signals (which are functions of s(t)) deteriorate first.
  - ~35% of episodes are TRANSIENT: s peaks briefly then decays, the loan
    registers 1–2 partial/missed payments and then CURES. These are the EWS's
    realistic false positives.
  - The rest are TERMINAL: s sustains high, misses accumulate (+30 DPD each),
    DPD rolls 30 → 60 → 90 (default_flag = 1, sticky) → possibly 180 (written
    off). Personal loans may cure after 90+ but default_flag stays 1.

Per-month payment outcome: P(miss) = clip(baseline_floor + gain·s(t)).
  baseline_floor = HAZARD_BASELINE_MONTHLY[product] · exp(-γz) · seasoning · macro
  → a few non-stress baseline defaults; stress drives the cascades.

Outputs (parquet checkpoints):  repayments, behaviour_monthly
The internal z is read here but never written to either panel.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    OBS_WINDOW_MONTHS, N_VINTAGES, VINTAGE_MACRO_HAZARD_SWING,
    SEASONING_PEAK_MOB, SEASONING_WIDTH, HAZARD_SEASONING_PEAK_MULT,
    HAZARD_BASELINE_MONTHLY, HAZARD_GAMMA,
    STRESS_BASELINE_ONSET_P, STRESS_ONSET_Z_SLOPE,
    STRESS_RAMP_MONTHS_RANGE, STRESS_PEAK_RANGE, STRESS_CURE_FRACTION,
    STRESS_TRANSIENT_DECAY_MONTHS, STRESS_TERMINAL_SUSTAIN_RANGE,
    MISS_GAIN_BY_PRODUCT, P_MISS_CAP, PARTIAL_SHARE_UNDER_STRESS, PARTIAL_DPD_STEP,
    DPD_PER_MISS, DEFAULT_DPD_THRESHOLD, WRITEOFF_DPD_THRESHOLD,
    BNPL_EARLY_MONTHS, BNPL_EARLY_MISS_P, BNPL_EARLY_GAMMA, POST_90_MISS_P,
    CURE_P_BY_BUCKET, CURE_P_BY_PRODUCT_BUCKET, POST_DEFAULT_CURE_P,
    PREPAY_MONTHLY_P,
    BEHAVIOUR_BASELINE, BEHAVIOUR_UNDER_STRESS, BEHAVIOUR_NOISE_SD,
    get_rng,
)

# --- calendar -----------------------------------------------------------------
# Vintage 1 = 2024-07, ... vintage 24 = 2026-06; snapshot = end of 2026-06.
VINTAGE_START = pd.Timestamp("2024-07-01")
SNAPSHOT_MONTH_IDX = OBS_WINDOW_MONTHS  # 24


def _month_idx(ts):
    """Calendar month -> 1..24 (2024-07 = 1)."""
    ts = pd.Timestamp(ts)
    return (ts.year - VINTAGE_START.year) * 12 + (ts.month - VINTAGE_START.month) + 1


def _bucket_from_dpd(dpd):
    if dpd <= 0:
        return "Current"
    if dpd <= 30:
        return "1-30"
    if dpd <= 60:
        return "31-60"
    if dpd <= 90:
        return "61-90"
    return "90+"


# -----------------------------------------------------------------------------
# stress process
# -----------------------------------------------------------------------------

def _build_stress_series(z, rng, n_months):
    """Per-customer monthly stress s(t) in [0,1] over calendar months 1..n.

    Returns (s, is_terminal) where is_terminal[t] flags months that belong to a
    terminal (default-capable) episode. Onset probability rises as z falls.
    """
    s = np.zeros(n_months + 1)            # 1-indexed; index 0 unused
    terminal = np.zeros(n_months + 1, dtype=bool)
    p_onset = STRESS_BASELINE_ONSET_P * np.exp(STRESS_ONSET_Z_SLOPE * (-z))
    p_onset = min(p_onset, 0.5)

    t = 1
    while t <= n_months:
        if s[t] == 0 and rng.random() < p_onset:
            ramp = int(rng.integers(STRESS_RAMP_MONTHS_RANGE[0], STRESS_RAMP_MONTHS_RANGE[1] + 1))
            peak = float(rng.uniform(*STRESS_PEAK_RANGE))
            is_term = rng.random() >= STRESS_CURE_FRACTION   # 65% terminal
            # ramp up
            k = 0
            while k < ramp and t + k <= n_months:
                s[t + k] = peak * (k + 1) / ramp
                k += 1
            cursor = t + ramp
            if is_term:
                sustain = int(rng.integers(STRESS_TERMINAL_SUSTAIN_RANGE[0],
                                           STRESS_TERMINAL_SUSTAIN_RANGE[1] + 1))
                for j in range(sustain):
                    if cursor + j <= n_months:
                        s[cursor + j] = peak
                        terminal[cursor + j] = True
                # mark the ramp months terminal too (they lead the default)
                for k in range(ramp):
                    if t + k <= n_months:
                        terminal[t + k] = True
                cursor += sustain
            else:  # transient: decay back to 0
                for j in range(STRESS_TRANSIENT_DECAY_MONTHS):
                    if cursor + j <= n_months:
                        s[cursor + j] = peak * (1 - (j + 1) / (STRESS_TRANSIENT_DECAY_MONTHS + 1))
                cursor += STRESS_TRANSIENT_DECAY_MONTHS
            t = cursor + 1
        else:
            t += 1
    return s, terminal


def _seasoning_mult(product, mob):
    """Product-specific hump on hazard, peaking at SEASONING_PEAK_MOB."""
    peak = SEASONING_PEAK_MOB[product]
    width = SEASONING_WIDTH[product]
    base = np.exp(-((mob - peak) / width) ** 2)
    return 1.0 + (HAZARD_SEASONING_PEAK_MULT[product] - 1.0) * base


# -----------------------------------------------------------------------------
# per-loan monthly simulation
# -----------------------------------------------------------------------------

def _simulate_loan(loan, z, s_series, macro_by_month, rng):
    """Walk one loan month by month. Returns list of repayment-row dicts."""
    product = loan["product_type"]
    grade = loan["origination_risk_grade"]
    ticket = float(loan["ticket_size"])
    tenure = int(loan["tenure_mo"])
    apr = float(loan["interest_rate_apr"])
    orig = pd.Timestamp(loan["origination_date"])
    orig_idx = _month_idx(orig)

    # amortisation: level EMI
    r = apr / 100.0 / 12.0
    if r > 0:
        emi = ticket * r / (1 - (1 + r) ** (-tenure))
    else:
        emi = ticket / tenure

    gain = MISS_GAIN_BY_PRODUCT[product]
    floor0 = HAZARD_BASELINE_MONTHLY[product] * np.exp(-HAZARD_GAMMA * z)
    prepay_p = PREPAY_MONTHLY_P[product][grade]

    rows = []
    outstanding = ticket
    dpd = 0
    missed_cum = 0
    defaulted = False
    status = "Active"

    for k in range(1, tenure + 1):
        cal_idx = orig_idx + (k - 1)
        if cal_idx > SNAPSHOT_MONTH_IDX:
            break  # right-censored at snapshot
        period_date = orig + pd.DateOffset(months=k - 1)

        s_t = s_series[cal_idx] if cal_idx <= len(s_series) - 1 else 0.0
        macro = macro_by_month[cal_idx]
        if product == "BNPL" and k <= BNPL_EARLY_MONTHS:
            # flat high impulse floor early in life (A-066 fast-roll)
            floor = min(BNPL_EARLY_MISS_P * np.exp(-BNPL_EARLY_GAMMA * z) * macro, 0.85)
        else:
            seas = _seasoning_mult(product, k)
            floor = min(floor0 * seas * macro, 0.20)
        p_miss = min(floor + gain * s_t, P_MISS_CAP)
        if defaulted:  # 90+ loans mostly charge off rather than cure (R11 loss realism)
            p_miss = max(p_miss, POST_90_MISS_P)

        # scheduled interest / principal split on the current balance
        interest_due = outstanding * r
        principal_due = max(emi - interest_due, 0.0)

        u = rng.random()
        if u < p_miss:
            outcome = "miss"
        elif u < p_miss + (1 - p_miss) * (PARTIAL_SHARE_UNDER_STRESS if s_t > 0.05 else 0.05):
            outcome = "partial"
        else:
            outcome = "full"

        # apply outcome to DPD / balance
        if outcome == "full":
            amount_paid = emi
            if dpd > 0:  # attempt cure
                bucket = _bucket_from_dpd(dpd)
                if dpd >= DEFAULT_DPD_THRESHOLD:
                    cure_p = POST_DEFAULT_CURE_P[product]
                else:
                    cure_p = CURE_P_BY_PRODUCT_BUCKET[product].get(bucket, 0.0)
                if rng.random() < cure_p:
                    dpd = 0
                else:
                    dpd = max(0, dpd - DPD_PER_MISS)
            outstanding = max(0.0, outstanding - principal_due)
        elif outcome == "partial":
            amount_paid = emi * rng.uniform(0.3, 0.7)
            dpd += PARTIAL_DPD_STEP
            paid_principal = max(amount_paid - interest_due, 0.0)
            outstanding = max(0.0, outstanding - paid_principal * 0.5)
        else:  # miss
            amount_paid = 0.0
            dpd += DPD_PER_MISS
            # unpaid interest capitalises slightly
            outstanding = outstanding + interest_due * 0.5

        if dpd >= DEFAULT_DPD_THRESHOLD and not defaulted:
            defaulted = True

        bucket = _bucket_from_dpd(dpd)
        if outcome != "full":
            missed_cum += 1
        paid_on_time = int(outcome == "full" and dpd == 0)
        partial_flag = int(outcome == "partial")

        write_off = dpd >= WRITEOFF_DPD_THRESHOLD
        # prepayment only from a performing (Current) state
        prepay = (not write_off) and dpd == 0 and outstanding > 0 and rng.random() < prepay_p

        rows.append({
            "loan_id": loan["loan_id"],
            "customer_id": loan["customer_id"],
            "period_index": k,
            "period_date": period_date,
            "period_month_idx": cal_idx,
            "scheduled_emi": round(emi, 2),
            "outstanding_balance": round(max(outstanding, 0.0), 2),
            "amount_paid": round(amount_paid, 2),
            "dpd": int(dpd),
            "delinquency_bucket": bucket,
            "paid_on_time_flag": paid_on_time,
            "partial_payment_flag": partial_flag,
            "missed_payments_cum": missed_cum,
            "prepayment_flag": int(prepay),
            "default_flag": int(defaulted),
        })

        if write_off:
            status = "Written-off"
            break
        if prepay:
            status = "Closed"
            break
        if outstanding <= 1.0 and k >= 1:
            status = "Closed"
            break
        if k == tenure:
            status = "Active" if cal_idx >= SNAPSHOT_MONTH_IDX else "Closed"

    # final status: if last observed period < tenure and not terminated -> Active (censored)
    if rows:
        last = rows[-1]
        if status not in ("Written-off", "Closed"):
            status = "Active"
    return rows, status, int(defaulted)


# -----------------------------------------------------------------------------
# behaviour signals
# -----------------------------------------------------------------------------

def _behaviour_rows(customer_id, z, active_months, s_series, period_date_by_idx, rng):
    """One row per active customer-month, signals = lerp(baseline, stress, s)."""
    base = BEHAVIOUR_BASELINE
    strs = BEHAVIOUR_UNDER_STRESS
    rows = []
    for cal_idx in sorted(active_months):
        s_t = s_series[cal_idx] if cal_idx <= len(s_series) - 1 else 0.0

        def lerp(key):
            return base[key] * (1 - s_t) + strs[key] * s_t

        noise = lambda: rng.normal(1.0, BEHAVIOUR_NOISE_SD)
        cf = np.clip(lerp("cashflow_consistency") * noise() + 0.03 * z, 0.0, 1.0)
        bv = max(0.0, lerp("balance_volatility") * noise() - 0.02 * z)
        shock = int(rng.random() < lerp("spending_shock_p"))
        nach = int(rng.poisson(lerp("nach_bounce_rate")))
        util = np.clip(lerp("credit_utilisation") * noise(), 0.0, 1.5) * 100
        inq = int(rng.poisson(lerp("bureau_inquiry_rate_90d")))
        app = max(0.0, lerp("app_engagement") * noise())

        rows.append({
            "customer_id": customer_id,
            "month": period_date_by_idx[cal_idx].strftime("%Y-%m"),
            "month_idx": cal_idx,
            "cashflow_consistency": round(float(cf), 4),
            "balance_volatility": round(float(bv), 4),
            "spending_shock_flag": shock,
            "nach_bounce_count": min(nach, 5),
            "credit_utilisation": round(float(util), 2),
            "bureau_inquiry_velocity": min(inq, 10),
            "app_engagement": round(float(app), 2),
        })
    return rows


# -----------------------------------------------------------------------------
# main driver
# -----------------------------------------------------------------------------

def run_engine(customers_df, loans_df):
    z_by_cust = dict(zip(customers_df["customer_id"], customers_df["true_latent_risk"]))
    loans_by_cust = {cid: g for cid, g in loans_df.groupby("customer_id", sort=True)}

    macro_rng = get_rng("engine", "macro")
    macro_by_month = {m: 1.0 + macro_rng.uniform(-VINTAGE_MACRO_HAZARD_SWING, VINTAGE_MACRO_HAZARD_SWING)
                      for m in range(1, OBS_WINDOW_MONTHS + 1)}

    rep_rows, beh_rows = [], []
    loan_status, loan_default = {}, {}

    cust_ids = sorted(loans_by_cust.keys())
    for ci, cid in enumerate(cust_ids):
        z = float(z_by_cust[cid])
        rng = get_rng("engine", cid)
        s_series, _term = _build_stress_series(z, rng, OBS_WINDOW_MONTHS)

        cust_loans = loans_by_cust[cid].sort_values("loan_sequence")
        active_months = set()
        period_date_by_idx = {}
        for _, loan in cust_loans.iterrows():
            rows, status, deflag = _simulate_loan(loan, z, s_series, macro_by_month, rng)
            rep_rows.extend(rows)
            loan_status[loan["loan_id"]] = status
            loan_default[loan["loan_id"]] = deflag
            for rr in rows:
                active_months.add(rr["period_month_idx"])
                period_date_by_idx[rr["period_month_idx"]] = pd.Timestamp(rr["period_date"]).replace(day=1)

        beh_rows.extend(_behaviour_rows(cid, z, active_months, s_series, period_date_by_idx, rng))

        if (ci + 1) % 8000 == 0:
            print(f"  ... {ci+1:,}/{len(cust_ids):,} customers simulated")

    repayments = pd.DataFrame(rep_rows)
    behaviour = pd.DataFrame(beh_rows)
    return repayments, behaviour, loan_status, loan_default


# -----------------------------------------------------------------------------
# report + spot-check (behaviour leads default)
# -----------------------------------------------------------------------------

def report(repayments, behaviour, loans_df, loan_status, loan_default):
    print(f"\nrepayments rows: {len(repayments):,}   (target ~550k, ±10% = 495k–605k)")
    print(f"behaviour rows : {len(behaviour):,}   (target ~480k, ±10% = 432k–528k)")

    # loan-level default
    dser = pd.Series(loan_default)
    sser = pd.Series(loan_status)
    loans = loans_df.set_index("loan_id")
    loans = loans.assign(default_flag=dser, loan_status=sser)
    blended = loans["default_flag"].mean()
    print(f"\nBlended 90+ default: {blended:.2%}   (target ~7.5%, range 6–9%)")
    print("By product:")
    for p, g in loans.groupby("product_type", observed=True):
        print(f"    {p:<10} {g['default_flag'].mean():.2%}   n={len(g):,}")
    print("By grade (should rise A→E):")
    for gr in ["A", "B", "C", "D", "E"]:
        g = loans[loans["origination_risk_grade"] == gr]
        if len(g):
            print(f"    {gr}  {g['default_flag'].mean():.2%}   n={len(g):,}")
    print("loan_status mix:")
    print(sser.value_counts(normalize=True).to_string())

    # ---- spot-check: behaviour leads default -------------------------------
    print("\n=== LEAD-TIME SPOT CHECK (behaviour must lead the 90+ event) ===")
    # for each defaulted loan, find first period it hit 90+ (calendar idx),
    # then find the earliest month in the preceding 4 where the customer's
    # behaviour had already deteriorated (cashflow below a stress threshold).
    rep = repayments
    first90 = (rep[rep["dpd"] >= 90]
               .sort_values(["loan_id", "period_index"])
               .groupby("loan_id").first()[["customer_id", "period_month_idx"]])
    beh_idx = behaviour.set_index(["customer_id", "month_idx"])["cashflow_consistency"]
    leads = []
    sample = first90.sample(min(2000, len(first90)), random_state=1) if len(first90) else first90
    for loan_id, row in sample.iterrows():
        cid, m90 = row["customer_id"], int(row["period_month_idx"])
        onset = None
        for back in range(1, 7):
            mi = m90 - back
            if (cid, mi) in beh_idx.index:
                if float(beh_idx.loc[(cid, mi)]) < 0.60:   # deteriorated
                    onset = mi
        if onset is not None:
            leads.append((m90 - onset) * 30)  # months -> ~days
    if leads:
        leads = np.array(leads)
        print(f"  defaulted loans sampled: {len(sample):,}; with detectable prior deterioration: {len(leads):,}")
        print(f"  median lead: {np.median(leads):.0f} days   (Gate-1 needs >= 60)   "
              f"{'PASS' if np.median(leads) >= 60 else 'FAIL'}")
        print(f"  mean lead:   {leads.mean():.0f} days")
    else:
        print("  no leads measured (check stress wiring)")

    # cure-fraction proxy: loans that touched delinquency but never defaulted
    touched = rep.groupby("loan_id")["dpd"].max()
    delinq = touched[touched > 0]
    cured_no_default = (delinq < 90).mean() if len(delinq) else float("nan")
    print(f"\nDelinquent-but-never-90+ share (false-positive proxy): {cured_no_default:.1%}   (design ~35%)")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    pq = root / "out" / "data" / "parquet"
    customers_df = pd.read_parquet(pq / "customers.parquet")
    loans_df = pd.read_parquet(pq / "loans.parquet")
    print(f"Loaded {len(customers_df):,} customers, {len(loans_df):,} loans\n")
    print("Simulating monthly panels (this is the heavy step)...")

    repayments, behaviour, loan_status, loan_default = run_engine(customers_df, loans_df)
    report(repayments, behaviour, loans_df, loan_status, loan_default)

    repayments.to_parquet(pq / "repayments.parquet", index=False)
    behaviour.to_parquet(pq / "behaviour_monthly.parquet", index=False)
    # persist the loan-level outcomes the engine produced, for Step 5
    pd.Series(loan_status).rename("loan_status").to_frame().to_parquet(pq / "_loan_status.parquet")
    pd.Series(loan_default).rename("default_flag").to_frame().to_parquet(pq / "_loan_default.parquet")
    print(f"\nWritten: repayments.parquet, behaviour_monthly.parquet (+ interim outcome files)")
