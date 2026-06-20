"""
config.py — The single rulebook for the Early Warning System (Workstream F).

Every choice the EWS depends on lives here, organised by build step. Nothing
else in the codebase should hard-code a column name, a window length, a
threshold or a path; everything imports from here.

Two kinds of value live in this file (same convention as WS B/C/E):

  1. LOCKED  — fixed by the data spec, the E->F handoff memo, or a logged
     assumption. Authoritative; do not touch without a logged decision.

  2. TUNE    — starting values for knobs we calibrate in later steps
     (label window, model hyper-parameters, RAG band cut-offs). Each is
     marked TUNE and has a "settled in Step N" note.

The one discipline that overrides convenience everywhere: POINT-IN-TIME.
To score month t, only rows dated <= t may be read. See section 5.

References: WS_F_handoff_memo.md (memo §n), Assumptions Log (A-0xx),
Workstream C data spec (§n), Gate 4 acceptance test.
"""

import hashlib
from pathlib import Path
from numpy.random import SeedSequence, default_rng


# =============================================================================
# 1. Reproducibility — seed + RNG helper                        (A-062, memo §6)
# =============================================================================
# Reuse the project master seed so F's splits/models are reproducible and sit
# in the same lineage as the data and segmentation. Same seed -> same scores.

MASTER_SEED = 20260603
WORKSTREAM = "F"
EWS_VERSION = "0.1-draft"


def _key_to_int(k):
    """Hash any string/int to a stable 32-bit integer (deterministic)."""
    if isinstance(k, int):
        return k & 0xFFFFFFFF
    h = hashlib.blake2b(str(k).encode(), digest_size=4).digest()
    return int.from_bytes(h, "big")


def get_rng(*stream_keys):
    """
    Return a numpy Generator seeded from MASTER_SEED + the given keys, so each
    named sub-stream (split, bootstrap, model init) is independent yet stable.

        get_rng("split", "oot")
        get_rng("bootstrap", iteration_id)
    """
    spawn_key = tuple(_key_to_int(k) for k in stream_keys)
    return default_rng(SeedSequence(MASTER_SEED, spawn_key=spawn_key))


# =============================================================================
# 2. Paths                                                             (memo §5)
# =============================================================================
# Inputs are the WS-C source tables + the frozen WS-E segment assignment table.
# Adjust ROOT to wherever the WS_E_FINAL_CLEAN package is unpacked.

ROOT = Path("inputs")                       # WS-C tables (LOCKED inputs)
SEG_DIR = Path("FINAL")                     # WS-E canonical deliverable
VALID_DIR = Path("validation_only")         # audit-only, never features
OUT = Path("out")                           # F's outputs

PATHS = {
    "customers":   ROOT / "customers.parquet",
    "loans":       ROOT / "loans.parquet",
    "repayments":  ROOT / "repayments.parquet",
    "behaviour":   ROOT / "behaviour_monthly.parquet",
    "segments":    SEG_DIR / "E_final_segment_assignments.csv",
    # AUDIT ONLY — creditworthiness latent (higher = safer). NEVER a feature.
    "true_latent": VALID_DIR / "_true_latent_risk_VALIDATION_ONLY.parquet",   # A-019
}

OUT_DIRS = {
    "scores":     OUT / "scores",
    "triggers":   OUT / "triggers",
    "thresholds": OUT / "thresholds",
    "validation": OUT / "validation",
    "gate4":      OUT / "gate4_results",
}


# =============================================================================
# 3. Schema constants — column names & categorical domains   (WS-C data spec §5)
# =============================================================================
# Verified against the shipped tables. Reference these, never literal strings.

KEYS = {"loan": "loan_id", "customer": "customer_id"}

# behaviour_monthly — F's primary raw material (memo §1.3). Panel grain:
# one row per customer x calendar month. `month` is "YYYY-MM".
BEHAVIOUR_MONTH_COL = "month"
BEHAVIOUR_SIGNALS = [
    "cashflow_consistency",     # higher = steadier income          (stress: falls)
    "balance_volatility",       # higher = choppier balance         (stress: spikes)
    "credit_utilisation",       # % of limit drawn                  (stress: creeps up)
    "nach_bounce_count",        # auto-debit bounces in the month   (stress: streaks)
    "spending_shock_flag",      # 0/1 sudden spend dislocation      (stress: onset)
    "bureau_inquiry_velocity",  # credit-hunger signal              (stress: accelerates)
    "app_engagement",           # softer signal                     (ambiguous)
]

# repayments — the DPD / default timeline (the OUTCOME source). Grain:
# one row per loan x scheduled period. `period_date` is the calendar anchor.
REPAY_PERIOD_DATE = "period_date"
REPAY_COLS = {
    "dpd": "dpd",                               # 0..195 observed
    "bucket": "delinquency_bucket",             # Current/1-30/31-60/61-90/90+
    "outstanding": "outstanding_balance",
    "partial_flag": "partial_payment_flag",
    "on_time_flag": "paid_on_time_flag",
    "missed_cum": "missed_payments_cum",
    "prepay_flag": "prepayment_flag",
}
DELQ_BUCKETS_ORDERED = ["Current", "1-30", "31-60", "61-90", "90+"]

# loans — origination + outcome
LOAN_COLS = {
    "product": "product_type",                  # Personal / BNPL / SME
    "grade": "origination_risk_grade",          # A..E
    "ticket": "ticket_size",
    "tenure": "tenure_mo",                       # 1..48
    "apr": "interest_rate_apr",
    "orig_date": "origination_date",
    "cohort": "origination_cohort",              # "YYYY-MM"
    "mob": "months_on_book",                     # 1..24
    "status": "loan_status",                     # Active / Closed / Written-off
    "default_flag": "default_flag",              # ever 90+ (sticky), loan-level
    "value_proxy": "value_proxy",                # WS-C frozen; F never touches value
}

DOMAINS = {
    "product_type": ["BNPL", "Personal", "SME"],
    "grade": ["A", "B", "C", "D", "E"],
    "loan_status": ["Active", "Closed", "Written-off"],
}

# Calendar span of the dataset (LOCKED — from shipped tables):
#   origination & repayment periods run 2024-07 .. 2026-06 (24 monthly cohorts).
DATA_START_MONTH = "2024-07"
DATA_END_MONTH = "2026-06"


# =============================================================================
# 4. Label definition — forward-window 90+                 (memo §2, §3; Step 3)
# =============================================================================
# F predicts whether a PERFORMING loan transitions to default soon.

DEFAULT_DPD = 90                 # LOCKED: default = first time DPD reaches 90+ (sticky, spec §9.1)
PERFORMING_MAX_DPD = 89          # LOCKED: "performing" at month t = DPD < 90 at start of t

# Seasoned basis: a loan is only eligible from MOB >= 3 (mechanically can't be
# 90+ before then). Same convention E used for default rates.
MOB_FLOOR = 3                    # LOCKED (memo §2 "seasoning/censoring")

# Forward window H: label = does the loan hit 90+ within months t+1 .. t+H.
# 3 months ~= the 60-90 day horizon the brief calls for. TUNE in Step 7
# (sensitivity-test H in {2,3,4}); 3 is the starting value.
LABEL_WINDOW_H = 3               # TUNE — settled in Step 7

# Censoring rule (LOCKED, memo §2 + Step 3): if a loan's panel ends (snapshot /
# prepay / closure) inside the t+1..t+H window with no observed 90+, the outcome
# is UNKNOWN -> DROP the row. Never code unknown as 0 (that fabricates "good"
# loans and biases risk down — the survivorship illusion the memo warns of).
DROP_CENSORED_LABELS = True

# Consequence of H against DATA_END_MONTH: the last observation month with a
# COMPLETE forward window is 2026-03 (its t+1..t+3 = Apr/May/Jun all observed).
# Observation months 2026-04..06 have truncated windows -> dropped in Step 3.
LAST_FULL_WINDOW_MONTH = "2026-03"   # derived from DATA_END_MONTH - H


# =============================================================================
# 5. Point-in-time rule & feature windows               (memo rule #1; Step 4)
# =============================================================================
# THE non-negotiable. A feature for month t may read behaviour/repayment rows
# with period_date / month <= t ONLY. Every feature is routed through one
# windowing helper (frame/features.py) that enforces this; nothing reads ahead.

POINT_IN_TIME = True             # LOCKED — do not disable
FEATURE_MAX_LAG_MONTH = "t"      # features observable strictly at or before t

# Rolling windows for trend/delta/streak features (memo §3 "trends and deltas,
# not just levels"). Lengths in months.
TREND_WINDOW = 3                 # rolling 3-month change (e.g. cashflow_consistency slope)
VOLATILITY_BASELINE_WINDOW = 6   # trailing baseline a spike is measured against
STREAK_LOOKBACK = 6              # max months back to count consecutive bounces/partials


# =============================================================================
# 6. Feature spec — what to build, and its monotone direction     (Step 4 + 6)
# =============================================================================
# Each engineered feature names its source signal, its transform, and the sign
# of its expected relationship to default risk. The sign feeds the MONOTONIC
# constraint on the GBM (Step 6) so the model can't learn an indefensible shape.
#   +1 : higher feature value -> higher risk
#   -1 : higher feature value -> lower risk
# C4 drivers (the behavioural-distress prototype, memo §1.2) are listed FIRST.

FEATURE_SPEC = [
    # name                          source                       transform            mono
    ("bounce_streak",               "nach_bounce_count",         "consecutive>0",     +1),
    ("bounce_count_3m",             "nach_bounce_count",         "rolling_sum_3m",    +1),
    ("spending_shock_recency",      "spending_shock_flag",       "months_since_last", -1),
    ("spending_shock_rate_3m",      "spending_shock_flag",       "rolling_mean_3m",   +1),
    ("utilisation_level",           "credit_utilisation",        "level",             +1),
    ("utilisation_creep_3m",        "credit_utilisation",        "delta_3m",          +1),
    ("balance_volatility_level",    "balance_volatility",        "level",             +1),
    ("cashflow_consistency_level",  "cashflow_consistency",      "level",             -1),
    ("cashflow_consistency_slope",  "cashflow_consistency",      "slope_3m",          -1),
    # rest of the panel
    ("inquiry_velocity_accel",      "bureau_inquiry_velocity",   "delta_3m",          +1),
    ("app_engagement_drop",         "app_engagement",            "delta_3m",          -1),
    # repayment-state, still LEADING (precede 90+): partial runs, first bounce
    ("partial_run",                 "partial_payment_flag",      "consecutive>0",     +1),
    ("first_bounce_flag",           "nach_bounce_count",         "first_occurrence",  +1),
]

# Coincident / escalation signals — BUILT and carried in the matrix, but
# EXCLUDED from the early-warning model. A loan that is already delinquent does
# not need an *early* warning; it needs collections. These drive the escalation
# trigger (TRIGGER_RULES below), not the score.
#   Step-6 finding: including dpd_current / missed_cum_at_t gave AUC 0.97 with
#   top flags 100% already 30+ DPD — near-coincident, zero lead time. Removing
#   them yields the honest behaviour-led model (~0.86 AUC on currently-clean
#   loans) that actually fires early.
ESCALATION_SPEC = [
    ("dpd_current",     "dpd",                 "level_at_t", +1),
    ("missed_cum_at_t", "missed_payments_cum", "level_at_t", +1),
]


# =============================================================================
# 7. Discrete trigger layer                                  (memo §3; Step 4)
# =============================================================================
# Rule-based flags that run ALONGSIDE the model so coverage doesn't depend on
# it alone. Conservative starting rules; TUNE precision/volume in Step 8.

TRIGGER_RULES = {
    "two_partials_plus_bounce":  {"partial_run": 2, "bounce_in_month": 1},  # memo §3 example
    "bounce_streak_2":           {"bounce_streak": 2},
    "shock_plus_util_creep":     {"spending_shock_flag": 1, "utilisation_creep_3m_pct": 10},
    "already_delinquent_30dpd":  {"dpd_current": 30},   # ESCALATION (not early-warning):
}  # TUNE thresholds — settled in Step 8        # routes already-late loans straight to collections


# =============================================================================
# 8. Splits — out-of-time is the decisive test               (memo §2; Step 5)
# =============================================================================
# Two splits. WITHIN-TIME (random, stratified) for tuning; OUT-OF-TIME (train
# on older observation months, test on newer) is the proof of generalisation
# the backtest reviewer required. A loan's months stay on ONE side of every
# split (group by loan_id) so nothing leaks across train/test.

SPLIT_GROUP_KEY = "loan_id"                 # LOCKED — group, never row-level
STRATIFY_KEYS = ["label", "segment"]        # within-time split stratification
WITHIN_TIME_TEST_FRAC = 0.25                # TUNE (minor)

# Out-of-time cut by OBSERVATION month: rows with period_date >= this go to the
# OOT test set; earlier rows train. Default leaves the 3 latest full-window
# months (Jan/Feb/Mar 2026) as a clean forward holdout. TUNE in Step 5.
OOT_TEST_FROM_MONTH = "2026-01"             # TUNE — settled in Step 5


# =============================================================================
# 9. Models                                                         (Step 6)
# =============================================================================
# Scorecard/logistic is the PRIMARY (governable, explainable per flag). The
# monotonic GBM is a challenger; it only replaces the scorecard if it beats it
# on out-of-time LEAD TIME *and* lift (accuracy that can't be governed is not
# an upgrade — memo §3, Step 6).

PRIMARY_MODEL = "scorecard"                 # TUNE — confirmed in Step 6 readout

SCORECARD = {
    "binning": "woe",                       # weight-of-evidence bins
    "max_bins": 6,
    "min_bin_frac": 0.05,
}

GBM = {                                     # TUNE — all hyper-params, Step 6/7
    "library": "lightgbm",
    "n_estimators": 400,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    # monotone constraints are read from FEATURE_SPEC signs at fit time.
    "use_monotone_from_feature_spec": True,
}

CALIBRATE_SCORES = True                     # calibrate to probabilities so §11 bands mean something


# =============================================================================
# 10. Validation targets — Gate 4                          (memo §4; Step 7,10)
# =============================================================================
# Gate 4 = PASS only if ALL four hold. Numeric targets below are acceptance
# guides; the lead-time and leakage criteria are pass/fail in spirit.

GATE4 = {
    "lead_time_days_min": 30,        # flags must fire meaningfully BEFORE 90+ (TUNE floor)
    "oot_auc_min": 0.70,             # honest out-of-time discrimination (guide)
    "oot_ks_min": 0.30,              # guide
    "top_decile_lift_min": 2.5,      # guide
    "stable_across_cohorts": True,   # performance holds across vintages & resamples
    "no_leakage": True,              # point-in-time + out-of-time audit clean
}

# Honesty constraint, not a target to beat: ~35% of stress episodes self-resolve
# by design (A-046), so some amber flags WILL cure without defaulting. This sets
# the achievable-precision ceiling; design bands around it, don't optimise it away.
SELF_CURE_EPISODE_RATE = 0.35        # LOCKED context (A-046)
BOOTSTRAP_RESAMPLES = 200            # stability check (Step 7)


# =============================================================================
# 11. RAG bands — turning the score into actions                  (Step 8)
# =============================================================================
# Green / Amber / Red on the calibrated probability. Cut-offs are TUNE: set them
# against (a) collections capacity and (b) the false-positive vs missed-default
# cost asymmetry — NOT round numbers. Starting guesses below.

RAG_BANDS = {                        # TUNE — settled in Step 8
    "red_min_prob": 0.40,            # proactive intervention tier
    "amber_min_prob": 0.15,          # watch tier
    # below amber_min_prob = green
}

# Operational constraints the band cut-offs must respect:
COLLECTIONS_MONTHLY_CAPACITY = None  # FILL: max red+amber flags the team can work / month
COST_FALSE_POSITIVE = None           # FILL: cost of chasing a loan that would have cured
COST_MISSED_DEFAULT = None           # FILL: cost of a 90+ we failed to flag in time
# -> the red cut is where (lead time x precision) justifies the FP cost; amber
#    fills remaining capacity. These three numbers come from the Risk Consultant.


# =============================================================================
# 12. Segments — connecting F back to E                     (memo §3; Step 9)
# =============================================================================
# Flag rates are reported against E's frozen 6-segment scheme. Two segments E
# routed explicitly to monitoring get priority in reporting.

SEGMENT_COL = "segment"
SEGMENTS = [
    "1. BNPL (grade A-C)",
    "2. Personal Prime (A-B)",
    "3. Personal Non-prime (C-E)",
    "4. SME Prime (A-B)",
    "5. SME Non-prime (C-E)",
    "6. Subprime BNPL (D-E)",
]
PRIORITY_SEGMENTS = [
    "3. Personal Non-prime (C-E)",   # Contain  — "route to early-warning monitoring"
    "6. Subprime BNPL (D-E)",        # Exit     — 44% default, wind-down watch
]

# The behavioural-distress cohort (E's ~66%-default "C4" group, defined by
# bounces/shocks/utilisation) is surfaced as the TOP watchlist — it is the
# prototype F operationalises into a live, leakage-clean score (memo §1.2).
SURFACE_BEHAVIOURAL_DISTRESS_AT_TOP = True


# =============================================================================
# 13. One-number-one-source                                (project principle)
# =============================================================================
# F's headline figures (monthly flag volume, precision, average lead time) are
# written to the controlled workbook so report/deck/dashboard read one source.
CONTROLLED_WORKBOOK_HEADLINES = ["flag_volume_monthly", "precision_at_threshold", "lead_time_days_mean"]
