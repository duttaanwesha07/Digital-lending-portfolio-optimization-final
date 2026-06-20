"""
config.py — The single rulebook for the synthetic dataset.

Every number from the Data Design Specification v1.0 and the Assumptions Log
lives here, organised by spec section. Nothing in the rest of the codebase
should hard-code a parameter; everything imports from here.

Two kinds of value live in this file:

  1. LOCKED values from the spec/log — mixes, ranges, targets. These are
     authoritative; we do not touch them.

  2. SOLVE-TO-TARGET starting values — internal knobs (z-weights, grade
     cut-points, hazard slope, seasoning shapes, stress parameters) that
     we tune in Step 7 until the locked targets hold within tolerance.
     Each one is clearly marked TUNE.
"""

import hashlib
from numpy.random import SeedSequence, default_rng


# =============================================================================
# 1. Reproducibility — seed + RNG helper                          (A-062, §11.1)
# =============================================================================
# One master seed for the whole run. Same seed → same dataset, byte-for-byte.
# A helper produces a fresh RNG for each named sub-stream so different parts
# of the pipeline never accidentally share or clobber each other's randomness.

MASTER_SEED = 20260603
SPEC_VERSION = "1.0"


def _key_to_int(k):
    """Hash any string/int to a stable 32-bit integer (deterministic across runs)."""
    if isinstance(k, int):
        return k & 0xFFFFFFFF
    h = hashlib.blake2b(str(k).encode(), digest_size=4).digest()
    return int.from_bytes(h, "big")


def get_rng(*stream_keys):
    """
    Return a numpy Generator seeded from MASTER_SEED + the given keys.

    Examples:
        get_rng("customers", "roots")              # one stream
        get_rng("loans", "ticket_size", "BNPL")    # nested stream
        get_rng("engine", customer_id)             # per-customer stream
    """
    spawn_key = tuple(_key_to_int(k) for k in stream_keys)
    ss = SeedSequence(MASTER_SEED, spawn_key=spawn_key)
    return default_rng(ss)


# =============================================================================
# 2. Volumes                                                    (§7.1, A-015)
# =============================================================================

N_CUSTOMERS = 40_000        # target customer count
TARGET_LOANS = 50_000       # target loan count (≈ 1.24 per customer)

# Repeat-borrower split: 80% hold 1 loan, 16% hold 2, 4% hold 3.
REPEAT_LOAN_SPLIT = {1: 0.80, 2: 0.16, 3: 0.04}


# =============================================================================
# 3. Customer root distributions                            (§7.2, A-027..A-031)
# =============================================================================
# These are the only fields drawn at random from their stated mix.

REGION_MIX = {"Urban": 0.45, "Semi-urban": 0.40, "Rural": 0.15}
INCOME_MIX = {"Low": 0.35, "Mid": 0.45, "High": 0.20}
EMPLOYMENT_MIX = {"Salaried": 0.45, "Self-employed": 0.25, "Gig": 0.18, "Informal": 0.12}

# Age: integer 18–65, right-skewed, median ≈ 32. We'll sample from a clipped
# lognormal in customers.py; the parameters are tuned to hit the median.
AGE_MIN, AGE_MAX = 18, 65
AGE_MEDIAN = 32

# Share of customers who are new-to-credit (thin file).
NEW_TO_CREDIT_SHARE = 0.35


# =============================================================================
# 4. Products — mix, ticket size, tenure                  (§7.3, A-032..A-034)
# =============================================================================

PRODUCT_MIX = {"BNPL": 0.35, "Personal": 0.45, "SME": 0.20}

# Ticket size — lognormal within each product, then clipped to the range.
# Median is the value you'd see most; sigma controls the spread.
TICKET = {
    "BNPL":     {"min":   2_000, "max":    50_000, "median":   12_000, "sigma": 0.85},
    "Personal": {"min":  30_000, "max":   500_000, "median":  120_000, "sigma": 0.70},
    "SME":      {"min": 200_000, "max": 5_000_000, "median":  800_000, "sigma": 0.85},
}

# Tenure in months — discrete within each product, peaked at the mode.
TENURE = {
    "BNPL":     {"min": 1, "max":  6, "mode":  3},
    "Personal": {"min": 6, "max": 36, "mode": 18},
    "SME":      {"min": 6, "max": 48, "mode": 24},
}

# Tighter terms for worse grades: worst grades get a modestly smaller
# ticket and shorter max tenure (R3). TUNE.
GRADE_TERM_SHRINK = {
    "A": {"ticket_factor": 1.00, "tenure_max_factor": 1.00},
    "B": {"ticket_factor": 1.00, "tenure_max_factor": 1.00},
    "C": {"ticket_factor": 0.95, "tenure_max_factor": 1.00},
    "D": {"ticket_factor": 0.85, "tenure_max_factor": 0.85},
    "E": {"ticket_factor": 0.70, "tenure_max_factor": 0.70},
}


# =============================================================================
# 5. Grade mix                                                  (§7.4, A-035)
# =============================================================================
# Target for the approved book. Grade cut-points (Section 14) are tuned to hit this.

GRADE_MIX = {"A": 0.15, "B": 0.30, "C": 0.30, "D": 0.18, "E": 0.07}
GRADES = ["A", "B", "C", "D", "E"]  # canonical order


# =============================================================================
# 6. Risk-based APR                                             (§7.5, A-036)
# =============================================================================
# For each product: a base APR (charged to grade A) and a premium per worse
# grade step. So an Personal-grade-C loan = 16% + 2 * 4pp = 24%.
# Plus a small Gaussian noise on top.

APR_BASE = {"BNPL": 24.0, "Personal": 16.0, "SME": 14.0}
APR_GRADE_STEP = {"BNPL": 3.0, "Personal": 4.0, "SME": 3.0}
APR_NOISE_SD = 0.5  # percentage-point standard deviation on APR
APR_MIN, APR_MAX = 12.0, 48.0


# =============================================================================
# 7. Channels — mix, CAC, TAT, z-tilt                     (§7.6, A-037..A-040)
# =============================================================================

CHANNELS = ["Digital ads", "Partner-embedded", "Referral", "DSA", "Organic"]

CHANNEL_MIX = {
    "Digital ads":      0.35,
    "Partner-embedded": 0.25,
    "Referral":         0.15,
    "DSA":              0.15,
    "Organic":          0.10,
}

# CAC central value (₹) per channel — lognormal around this with mild spread.
CHANNEL_CAC_MEDIAN = {
    "Digital ads":      1_200,
    "Partner-embedded":   600,
    "Referral":           300,
    "DSA":              1_000,
    "Organic":            150,
}
CHANNEL_CAC_SIGMA = 0.20  # log-sigma; gives modest noise around the median

# Approval TAT bands (hours) — lognormal within each band.
CHANNEL_TAT_BAND = {
    "Digital ads":      (2,  6),
    "Partner-embedded": (1,  4),
    "Referral":         (6, 24),
    "DSA":              (24, 72),
    "Organic":          (4, 24),
}

# z-tilt (standard-deviation units) — channel-of-acquisition shifts the
# selected customer's z slightly. Weak by design (§6.7); by-channel default
# differences are mostly composition.
CHANNEL_Z_TILT = {
    "Digital ads":      -0.10,
    "Partner-embedded":  0.00,
    "Referral":         +0.10,
    "DSA":              -0.15,
    "Organic":          +0.10,
}


# =============================================================================
# 8. Default targets                                       (§7.7, A-041..A-043)
# =============================================================================
# These are what the validator measures. Internal hazard knobs (Section 14)
# are tuned to hit them.

DEFAULT_TARGET_BLENDED = 0.075          # ~7.5%, acceptable 6–9% (A-041)

DEFAULT_TARGET_BY_PRODUCT = {           # (A-042)
    "BNPL":     0.10,
    "Personal": 0.07,
    "SME":      0.05,
}

DEFAULT_TARGET_BY_GRADE = {             # (A-043)
    "A": 0.015,
    "B": 0.035,
    "C": 0.07,
    "D": 0.14,
    "E": 0.26,
}


# =============================================================================
# 9. value_proxy inputs                                  (§7.8, A-044/48/49/50)
# =============================================================================

# Loss-given-default by product (A-044)
LGD_BY_PRODUCT = {"BNPL": 0.40, "Personal": 0.65, "SME": 0.75}

# Fee revenue as fraction of ticket, charged at origination (A-048).
# BNPL fee largely merchant-subsidised → low borrower-facing fee.
FEE_RATE_BY_PRODUCT = {"BNPL": 0.005, "Personal": 0.015, "SME": 0.015}

SERVICING_PER_ACTIVE_MONTH = 80.0       # ₹ per active loan-month (A-049)
COST_OF_FUNDS_ANNUAL = 0.11             # 11% annual (A-050)


# =============================================================================
# 10. Behavioural baselines & stress response             (§7.9, A-046, A-047)
# =============================================================================

# Normal-state baselines (mean values for a typical customer).
BEHAVIOUR_BASELINE = {
    "cashflow_consistency":     0.75,
    "balance_volatility":       0.20,
    "spending_shock_p":         0.03,
    "nach_bounce_rate":         0.05,   # Poisson rate, normal months
    "credit_utilisation":       0.35,
    "bureau_inquiry_rate_90d":  0.5,
    "app_engagement":           15.0,   # sessions/month, baseline
}

# Stress-state targets (what the signals look like at peak stress).
BEHAVIOUR_UNDER_STRESS = {
    "cashflow_consistency":     0.40,
    "balance_volatility":       0.65,
    "spending_shock_p":         0.40,
    "nach_bounce_rate":         2.0,
    "credit_utilisation":       0.90,
    "bureau_inquiry_rate_90d":  3.0,
    "app_engagement":           5.0,
}

# Fraction of stress episodes that resolve on their own without ever
# reaching 90+ DPD. Realistic false positives for the EWS (A-046).
STRESS_CURE_FRACTION = 0.35


# =============================================================================
# 11. Time, vintages, seasoning                          (§8.1–§8.3, A-051..A-053)
# =============================================================================

OBS_WINDOW_MONTHS = 24                   # M1..M24, snapshot at end of M24
N_VINTAGES = 24                          # one vintage per month
VINTAGE_GROWTH = 0.13                    # ~10–15% growth: newer cohorts larger
VINTAGE_MACRO_HAZARD_SWING = 0.02        # ±1–2pp macro factor on hazard

# Seasoning hump peak MOB per product (§8.3).
SEASONING_PEAK_MOB = {"BNPL": 2, "Personal": 8, "SME": 12}
SEASONING_WIDTH = {"BNPL": 1.3, "Personal": 4.0, "SME": 6.0}


# =============================================================================
# 12. Prepayment                                                       (§8.4)
# =============================================================================
# Per-month probability that an active, non-delinquent loan settles early.
# Higher for good grades; near zero for BNPL (short tenor anyway). TUNE.

PREPAY_MONTHLY_P = {
    "BNPL":     {"A": 0.005, "B": 0.004, "C": 0.003, "D": 0.002, "E": 0.001},
    "Personal": {"A": 0.025, "B": 0.020, "C": 0.015, "D": 0.010, "E": 0.005},
    "SME":      {"A": 0.020, "B": 0.015, "C": 0.010, "D": 0.005, "E": 0.0025},
}


# =============================================================================
# 13. DPD state machine                                        (§8.5, A-007/8)
# =============================================================================

DPD_PER_MISS = 30                # missed payment advances DPD by ~30
DEFAULT_DPD_THRESHOLD = 90       # ≥90 DPD → default_flag = 1 (sticky)
WRITEOFF_DPD_THRESHOLD = 180     # ≥180 DPD → written-off; panel stops

# Cure probability when a full payment is made from a delinquent state.
# Easier to cure from shallow buckets than deep ones.
CURE_P_BY_BUCKET = {"1-30": 0.80, "31-60": 0.50, "61-90": 0.30, "90+": 0.15}

# Per-product cure character (A-066): BNPL fast-roll / little cure window;
# Personal meaningful early cure; SME lumpy. Used in the engine. TUNE.
CURE_P_BY_PRODUCT_BUCKET = {
    "BNPL":     {"1-30": 0.15, "31-60": 0.06, "61-90": 0.03},
    "Personal": {"1-30": 0.82, "31-60": 0.55, "61-90": 0.28},
    "SME":      {"1-30": 0.45, "31-60": 0.30, "61-90": 0.18},
}


# =============================================================================
# 14. SOLVE-TO-TARGET starting values     [TUNE in Step 7 to hit §7 targets]
# =============================================================================
# These are NOT spec-locked. They are internal knobs we adjust to make the
# locked targets (grade mix, default rates, behaviour-leads-default lead time)
# hold within the Gate 1 tolerances.

# --- 14a. Latent z weights (§6.2) -------------------------------------------
# z = signed-weighted sum of demographic features + noise, standardised.
# Strong ≈ 0.30–0.40; Moderate ≈ 0.15–0.25; Weak ≈ 0.05–0.15.
# Sign convention: higher z = safer.

Z_WEIGHTS = {
    # income_proxy: High is safer
    "income": {"Low": -0.35, "Mid": 0.05, "High": +0.35},
    # employment: salaried safer, gig/informal riskier
    "employment": {"Salaried": +0.30, "Self-employed": +0.05, "Gig": -0.20, "Informal": -0.35},
    # region: urban safer
    "region": {"Urban": +0.20, "Semi-urban": 0.00, "Rural": -0.20},
    # new-to-credit: riskier (thin file)
    "new_to_credit": {0: 0.00, 1: -0.18},
    # age: prime (30–50) slightly safer; weak effect
    "age_effect_strength": 0.08,
    # idiosyncratic noise sd — picked so profile explains ~40–50% of z variance
    "noise_sd": 1.05,
}

# --- 14b. Grade cut-points (§6.3) -------------------------------------------
# z thresholds that bucket loans into A/B/C/D/E. TUNE to hit GRADE_MIX.
# Listed as the upper bounds for A, B, C, D (E gets the rest).
# Starting values assume z ≈ standard-normal: roughly the inverse CDF of the
# cumulative grade shares (0.15, 0.45, 0.75, 0.93).
GRADE_CUTPOINTS_Z = {"A": +1.04, "B": +0.13, "C": -0.67, "D": -1.48}
GRADE_UNDERWRITING_NOISE_SD = 0.25    # extra noise so origination_risk_grade ≠ credit_quality_indicator
BUREAU_NOISE_SD = 0.30                # for credit_quality_indicator
BUREAU_NOISE_SD_THINFILE = 0.50       # wider for new-to-credit

# --- 14c. Hazard model (§6.5) -----------------------------------------------
# h(t) = baseline(product, tenure) × exp(-γ·z) × seasoning × macro × stress_mult
# baseline calibrated so blended default lands near 7.5% and by-product
# matches; γ controls the by-grade gradient.

HAZARD_GAMMA = 1.02                  # slope on z — bigger = sharper grade ladder
HAZARD_BASELINE_MONTHLY = {          # baseline monthly hazard before modifiers
    "BNPL":     0.115,    # front-loaded impulse defaults (A-066), short tenor
    "Personal": 0.0060,
    "SME":      0.0030,
}
HAZARD_SEASONING_PEAK_MULT = {       # peak-of-hump multiplier on hazard
    "BNPL":     1.8,
    "Personal": 1.6,
    "SME":      1.5,
}
HAZARD_STRESS_PEAK_MULT = 5.0        # stress multiplier at s(t) = 1.0

# --- 14d. Stress process s(t) (§6.6) ----------------------------------------
# Per customer, per month, with prob increasing as z falls, a stress episode
# can start. If it starts: ramp up over 2–4 months to peak, then either
# resolve (~35% of episodes) or push toward 90+.

STRESS_BASELINE_ONSET_P = 0.0090       # baseline monthly prob of an episode at z=0
STRESS_ONSET_Z_SLOPE = 0.62            # how steeply onset prob rises as z falls
STRESS_RAMP_MONTHS_RANGE = (2, 4)     # ramp-up duration drawn uniformly here
STRESS_PEAK_RANGE = (0.6, 1.0)        # peak s value of an episode


# =============================================================================
# 15. Convenience — printable summary
# =============================================================================

def summary():
    """Print a quick human-readable summary; useful for sanity-checking."""
    lines = [
        f"Master seed:              {MASTER_SEED}",
        f"Spec version:             {SPEC_VERSION}",
        f"Customers / loans target: {N_CUSTOMERS:,} / {TARGET_LOANS:,}",
        f"Repeat split:             {REPEAT_LOAN_SPLIT}",
        f"Product mix:              {PRODUCT_MIX}",
        f"Grade mix target:         {GRADE_MIX}",
        f"Blended default target:   {DEFAULT_TARGET_BLENDED:.1%}",
        f"By-product default:       {DEFAULT_TARGET_BY_PRODUCT}",
        f"By-grade default:         {DEFAULT_TARGET_BY_GRADE}",
        f"LGD by product:           {LGD_BY_PRODUCT}",
        f"Servicing / month:        ₹{SERVICING_PER_ACTIVE_MONTH:.0f}",
        f"Cost of funds:            {COST_OF_FUNDS_ANNUAL:.1%}",
        f"Observation window:       {OBS_WINDOW_MONTHS} months",
        f"Hazard γ (TUNE):          {HAZARD_GAMMA}",
        f"Grade cut-points (TUNE):  {GRADE_CUTPOINTS_Z}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
    print()

    # Quick determinism check: same keys → same numbers; different keys → different.
    a1 = get_rng("customers", "roots").integers(0, 1_000_000, size=3)
    a2 = get_rng("customers", "roots").integers(0, 1_000_000, size=3)
    b  = get_rng("customers", "z_noise").integers(0, 1_000_000, size=3)
    print(f"Determinism check:")
    print(f"  stream 'customers/roots'    → {a1.tolist()}")
    print(f"  same stream again           → {a2.tolist()}   (must match)")
    print(f"  stream 'customers/z_noise'  → {b.tolist()}    (must differ)")
    assert (a1 == a2).all() and (a1 != b).any(), "RNG sub-stream isolation failed"
    print("  ✓ deterministic and isolated")


# =============================================================================
# 16. ENGINE DYNAMICS (Step 4)              [TUNE in Step 7 to hit §7 targets]
# =============================================================================
# These drive the monthly repayment/behaviour simulation. They implement the
# hazard form (A-024) and the behaviour-leads-default mechanism (A-025/§6.6):
# a per-customer stress process s(t) ramps up BEFORE the misses that cascade
# to 90+, so behaviour deteriorates first. ~35% of episodes are transient
# (short high-stress -> 1-2 misses -> cure) and act as the EWS false positives;
# the rest are terminal (sustained high-stress -> 3+ consecutive misses -> 90+).
# All are starting values; the calibration loop (Step 7) tunes them.

# Stress episode shape ---------------------------------------------------------
# Transient episode: after the ramp it decays back to zero over this many months
STRESS_TRANSIENT_DECAY_MONTHS = 2
# Terminal episode: after the ramp it SUSTAINS near-peak for this many months
# (long enough for DPD to roll 30 -> 60 -> 90 -> possibly 180), then decays.
STRESS_TERMINAL_SUSTAIN_RANGE = (4, 7)

# Payment-outcome model --------------------------------------------------------
# Monthly P(miss) = clip( baseline_floor(product,z,MOB) + gain(product)*s(t) ).
# The floor produces a few non-stress baseline defaults; stress drives cascades.
MISS_GAIN_BY_PRODUCT = {"BNPL": 0.95, "Personal": 0.88, "SME": 0.66}   # TUNE
P_MISS_CAP = 0.95
# Of the residual (not-a-miss) probability mass in a stressed month, the share
# that is a PARTIAL payment rather than a full payment (partials creep DPD).
PARTIAL_SHARE_UNDER_STRESS = 0.45        # TUNE
PARTIAL_DPD_STEP = 15                    # a partial advances DPD ~half a cycle
# Personal loans can cure even after touching 90+ (A-066); default_flag stays 1.
POST_DEFAULT_CURE_P = {"BNPL": 0.01, "Personal": 0.03, "SME": 0.02}    # TUNE
# Once 90+ (defaulted), most borrowers keep missing and roll toward 180/write-off
# (charge-off realism); makes realised loss bite so high-risk underperforms (R11). TUNE.
POST_90_MISS_P = 0.80

# Behaviour-signal lerp endpoints come from BEHAVIOUR_BASELINE / _UNDER_STRESS;
# small multiplicative noise on the continuous signals:
BEHAVIOUR_NOISE_SD = 0.16

# BNPL impulse front-loading (A-066): short tenor can't ride the slow-stress
# cascade, so BNPL gets a FLAT high early-miss floor over MOB 1-3 (z-modulated),
# producing fast-roll first-cycle defaults. TUNE.
BNPL_EARLY_MONTHS = 3
BNPL_EARLY_MISS_P = 0.27
BNPL_EARLY_GAMMA = 0.90
