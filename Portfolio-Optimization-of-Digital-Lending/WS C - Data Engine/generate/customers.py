"""
generate/customers.py — Step 2 of the build.

Produces the ~40,000-row customers table in causal order:

  1. Sample the demographic roots from the §7.2 mixes.
  2. Build the hidden risk score z from those roots + noise; standardise to N(0,1).
  3. Bucket (z + bureau noise) into credit grade A–E (§6.3).
  4. Assign acquisition_channel with the weak z-tilt (§6.7).
  5. Draw CAC from a lognormal around the channel median (§7.6).

The internal z (true_latent_risk) is kept in the table for validation only;
it gets dropped before the dataset is exported in Step 6 (A-019).

Note on grain: per A-017, acquisition_channel and cac are customer-grain;
approval_tat_hrs lives on the loan and is set in Step 3.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    N_CUSTOMERS,
    REGION_MIX, INCOME_MIX, EMPLOYMENT_MIX,
    AGE_MIN, AGE_MAX, AGE_MEDIAN, NEW_TO_CREDIT_SHARE,
    Z_WEIGHTS, GRADE_CUTPOINTS_Z, GRADES,
    BUREAU_NOISE_SD, BUREAU_NOISE_SD_THINFILE,
    CHANNELS, CHANNEL_MIX, CHANNEL_Z_TILT,
    CHANNEL_CAC_MEDIAN, CHANNEL_CAC_SIGMA,
    get_rng,
)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _choice(rng, mix_dict, n):
    """Sample n labels from a {label: prob} mix."""
    labels = list(mix_dict.keys())
    probs = list(mix_dict.values())
    return rng.choice(labels, size=n, p=probs)


def _sample_ages(rng, n):
    """Right-skewed integer ages in [18, 65] with median ≈ AGE_MEDIAN.

    Built as 18 + Gamma(shape, scale), clipped to [18, 65]. Shape/scale
    chosen so the median lands near AGE_MEDIAN; ages outside the range
    are resampled rather than clipped to avoid stacking mass at the bounds.
    """
    # Gamma(2.5, 6) has median ≈ 13.4, so 18 + that ≈ 31.4 — close to 32.
    out = np.empty(n, dtype=np.int32)
    filled = 0
    while filled < n:
        draws = 18 + rng.gamma(shape=2.5, scale=6.0, size=n - filled)
        draws = np.floor(draws).astype(np.int32)
        ok = (draws >= AGE_MIN) & (draws <= AGE_MAX)
        good = draws[ok]
        out[filled:filled + len(good)] = good
        filled += len(good)
    return out


def _z_contribution(values, weight_map):
    """Map an array of labels to their z-weight contributions."""
    lookup = np.vectorize(lambda v: weight_map[v])
    return lookup(values).astype(np.float64)


def _bucket_into_grades(z_observed):
    """Bucket z_observed into A/B/C/D/E using the cut-points in config."""
    cp = GRADE_CUTPOINTS_Z
    out = np.empty(len(z_observed), dtype="U1")
    out[z_observed >= cp["A"]] = "A"
    out[(z_observed >= cp["B"]) & (z_observed < cp["A"])] = "B"
    out[(z_observed >= cp["C"]) & (z_observed < cp["B"])] = "C"
    out[(z_observed >= cp["D"]) & (z_observed < cp["C"])] = "D"
    out[z_observed < cp["D"]] = "E"
    return out


def _assign_channels(z, rng):
    """Assign one of the 5 channels per customer, with a weak z-tilt.

    For a customer with latent risk z, the probability of channel c is
    proportional to base_mix[c] × exp(z_tilt[c] × z). Sampling uses the
    Gumbel-max trick (numerically equivalent to multinomial sampling with
    those probabilities, and fully vectorised).
    """
    n = len(z)
    log_mix = np.log([CHANNEL_MIX[c] for c in CHANNELS])
    tilts = np.array([CHANNEL_Z_TILT[c] for c in CHANNELS])
    # (n, K) matrix of per-(customer, channel) log-probabilities
    logits = log_mix[np.newaxis, :] + tilts[np.newaxis, :] * z[:, np.newaxis]
    gumbel = rng.gumbel(size=logits.shape)
    idx = np.argmax(logits + gumbel, axis=1)
    return np.array(CHANNELS)[idx]


def _sample_cac(channels, rng):
    """Per-customer CAC: lognormal around the channel's median."""
    medians = np.array([CHANNEL_CAC_MEDIAN[c] for c in channels])
    # lognormal: median = exp(mu), so mu = log(median)
    mu = np.log(medians)
    cac = rng.lognormal(mean=mu, sigma=CHANNEL_CAC_SIGMA)
    return np.round(cac).astype(np.int64)


# ----------------------------------------------------------------------------
# main builder
# ----------------------------------------------------------------------------

def make_customers():
    """Build the customers DataFrame. Deterministic given config.MASTER_SEED."""
    n = N_CUSTOMERS

    # ---- 1. demographic roots --------------------------------------------
    region    = _choice(get_rng("customers", "region"),     REGION_MIX,     n)
    income    = _choice(get_rng("customers", "income"),     INCOME_MIX,     n)
    employ    = _choice(get_rng("customers", "employment"), EMPLOYMENT_MIX, n)
    age       = _sample_ages(get_rng("customers", "age"), n)
    ntc       = get_rng("customers", "ntc").binomial(1, NEW_TO_CREDIT_SHARE, n).astype(np.int8)

    # ---- 2. latent risk z ------------------------------------------------
    z_income  = _z_contribution(income, Z_WEIGHTS["income"])
    z_employ  = _z_contribution(employ, Z_WEIGHTS["employment"])
    z_region  = _z_contribution(region, Z_WEIGHTS["region"])
    z_ntc     = _z_contribution(ntc.astype(int), Z_WEIGHTS["new_to_credit"])
    # Age contribution: prime (≈40) gets a small positive bump; tails get ~0.
    z_age     = Z_WEIGHTS["age_effect_strength"] * np.exp(-((age - 40) / 15.0) ** 2)
    z_signal  = z_income + z_employ + z_region + z_ntc + z_age
    z_noise   = get_rng("customers", "z_noise").normal(0.0, Z_WEIGHTS["noise_sd"], n)
    z_raw     = z_signal + z_noise
    # Standardise so z is ≈ N(0,1) across the population.
    z         = (z_raw - z_raw.mean()) / z_raw.std()

    # Track variance share explained by the demographic profile.
    var_signal = z_signal.var()
    var_total  = z_raw.var()
    profile_share = var_signal / var_total  # should be ~0.40–0.50

    # ---- 3. credit_quality_indicator -------------------------------------
    # Bureau is a noisy observation of z; wider noise for thin-file (new-to-credit).
    noise_sd = np.where(ntc == 1, BUREAU_NOISE_SD_THINFILE, BUREAU_NOISE_SD)
    bureau_noise = get_rng("customers", "bureau_noise").normal(0.0, 1.0, n) * noise_sd
    z_observed = z + bureau_noise
    credit_quality_indicator = _bucket_into_grades(z_observed)

    # ---- 4. acquisition_channel (z-tilt) --------------------------------
    channel = _assign_channels(z, get_rng("customers", "channel"))

    # ---- 5. cac ---------------------------------------------------------
    cac = _sample_cac(channel, get_rng("customers", "cac"))

    # ---- 6. customer_id -------------------------------------------------
    customer_id = np.array([f"C{i:06d}" for i in range(1, n + 1)])

    # ---- assemble -------------------------------------------------------
    df = pd.DataFrame({
        "customer_id":              customer_id,
        "region_type":              region,
        "income_proxy_band":        pd.Categorical(income, categories=list(INCOME_MIX.keys()), ordered=True),
        "employment_type":          employ,
        "age":                      age,
        "first_time_borrower":      ntc,
        "credit_quality_indicator": pd.Categorical(credit_quality_indicator, categories=GRADES, ordered=True),
        "acquisition_channel":      channel,
        "cac":                      cac,
        # internal-only; dropped on export in Step 6
        "true_latent_risk":         z.astype(np.float32),
    })

    # Attach a small "build report" to the frame for the runner to print.
    df.attrs["profile_share_of_z_variance"] = float(profile_share)
    return df


# ----------------------------------------------------------------------------
# report — prints what the manual asks us to eyeball
# ----------------------------------------------------------------------------

def _fmt_mix(series, target_dict, tol=0.03):
    """Pretty-print actual vs target mix with a tolerance flag."""
    actual = series.value_counts(normalize=True).sort_index()
    lines = []
    for label, target in target_dict.items():
        a = float(actual.get(label, 0.0))
        flag = "✓" if abs(a - target) <= tol else "✗"
        lines.append(f"    {label:<18} {a:6.1%}   (target {target:.0%}, ±{tol:.0%}) {flag}")
    return "\n".join(lines)


def report(df):
    n = len(df)
    z = df["true_latent_risk"].to_numpy()
    print(f"Customers built: {n:,}\n")

    print("Region mix:")
    print(_fmt_mix(df["region_type"], REGION_MIX))
    print("\nIncome mix:")
    print(_fmt_mix(df["income_proxy_band"], INCOME_MIX))
    print("\nEmployment mix:")
    print(_fmt_mix(df["employment_type"], EMPLOYMENT_MIX))
    print("\nNew-to-credit share:")
    ntc_actual = float(df["first_time_borrower"].mean())
    flag = "✓" if abs(ntc_actual - NEW_TO_CREDIT_SHARE) <= 0.03 else "✗"
    print(f"    {ntc_actual:6.1%}   (target {NEW_TO_CREDIT_SHARE:.0%}, ±3pp) {flag}")

    print("\nAge stats:")
    print(f"    min={int(df['age'].min())}, median={int(df['age'].median())}, "
          f"mean={df['age'].mean():.1f}, max={int(df['age'].max())}   (target median ≈ {AGE_MEDIAN})")

    print(f"\nLatent z stats (should be ≈ N(0,1)):")
    print(f"    mean={z.mean():+.4f}   std={z.std():.4f}   "
          f"min={z.min():+.2f}   max={z.max():+.2f}")
    pshare = df.attrs["profile_share_of_z_variance"]
    flag = "✓" if 0.40 <= pshare <= 0.50 else ("~" if 0.30 <= pshare <= 0.60 else "✗")
    print(f"    profile-share of variance: {pshare:.1%}   (target 40–50%) {flag}")

    print("\nGrade mix from credit_quality_indicator (channel/cac-free; later refined by underwriting):")
    from config import GRADE_MIX
    print(_fmt_mix(df["credit_quality_indicator"], GRADE_MIX))

    print("\nChannel mix:")
    print(_fmt_mix(df["acquisition_channel"], CHANNEL_MIX))

    print("\nMean z by channel (should match CHANNEL_Z_TILT direction):")
    z_by_ch = df.groupby("acquisition_channel", observed=True)["true_latent_risk"].mean()
    for ch in CHANNELS:
        tilt = CHANNEL_Z_TILT[ch]
        observed = float(z_by_ch[ch])
        # Direction check: observed should have same sign as tilt (when |tilt| > 0).
        dir_ok = "✓" if (tilt == 0) or (np.sign(observed) == np.sign(tilt)) else "✗"
        print(f"    {ch:<18} mean z = {observed:+.3f}   (designed tilt {tilt:+.2f}) {dir_ok}")

    print("\nCAC by channel:")
    cac_by_ch = df.groupby("acquisition_channel", observed=True)["cac"].median()
    for ch in CHANNELS:
        print(f"    {ch:<18} median ₹{int(cac_by_ch[ch]):>5,}   (target ₹{CHANNEL_CAC_MEDIAN[ch]:,})")

    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    df = make_customers()
    report(df)

    # Save as a parquet checkpoint so Step 3 can load it without regenerating.
    out_dir = Path(__file__).resolve().parent.parent / "out" / "data" / "parquet"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "customers.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nWritten: {out_path}")
