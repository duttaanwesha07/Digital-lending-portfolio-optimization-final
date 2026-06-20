"""
frame/features.py — Step 4: point-in-time behavioural features.

Turns the 7 raw signals into LEADING indicators — trends, deltas, streaks,
volatility — for every spine row. The one rule that governs everything here:

    A feature for month t may read only rows dated <= t.

We honour it structurally, not by hope: features are computed ON THE SOURCE
PANELS, where each row's value already depends only on its own past (every
operation is backward-looking — rolling / shift / cumulative; there is not a
single negative shift or forward reference in this file). The features are then
joined onto the spine by exact key, so the join cannot introduce look-ahead.

  - Behaviour features are CUSTOMER-level (panel: customer x month) -> exact
    join on (customer_id, obs_month). Coverage is 100% (checked).
  - Repayment features are LOAN-level (panel: loan x MOB) -> exact join on
    (loan_id, months_on_book).

`_true_latent_risk` is NEVER loaded in this file (A-019). It exists only for the
Step 7 audit.

What/where/direction for each feature is declared in config.FEATURE_SPEC; this
module is the implementation of those transforms, nothing more.

Run:  python -m frame.features
"""

import numpy as np
import pandas as pd

import config as cfg

EPS = 1e-9
W3 = cfg.TREND_WINDOW              # 3
W6 = cfg.VOLATILITY_BASELINE_WINDOW  # 6


# --- backward-only window helpers (operate within a sorted group) -----------

def _consecutive_positive(s: pd.Series, g: pd.Series) -> pd.Series:
    """Run-length of consecutive >0 values ending at each row, within group g."""
    pos = (s > 0).astype(int)
    # a new run starts wherever pos==0; cumsum of zeros gives a run id
    run_id = (pos == 0).groupby(g).cumsum()
    return pos.groupby([g, run_id]).cumsum()


def _months_since_last(flag: pd.Series, g: pd.Series, cap: int = 24) -> pd.Series:
    """Months since the last flag==1 at or before t (0 if flagged at t).
    Never-flagged -> cap (treated as 'long ago', the safe direction)."""
    pos = g.groupby(g).cumcount()                  # 0,1,2,... within customer
    last = pos.where(flag == 1)
    last = last.groupby(g).ffill()
    rec = (pos - last)
    return rec.fillna(cap).clip(upper=cap)


def _first_occurrence(s: pd.Series, g: pd.Series) -> pd.Series:
    """1 in the month a customer's signal first turns positive, else 0."""
    pos = (s > 0).astype(int)
    cum = pos.groupby(g).cumsum()
    return ((pos == 1) & (cum == 1)).astype(int)


def _roll(s, g, w, fn):
    """grouped backward rolling (min_periods=1) — current + prior w-1 rows."""
    return s.groupby(g).transform(lambda x: getattr(x.rolling(w, min_periods=1), fn)())


# --- behaviour (customer-level) features ------------------------------------

def behaviour_features() -> pd.DataFrame:
    b = pd.read_parquet(cfg.PATHS["behaviour"])
    b = b.sort_values([cfg.KEYS["customer"], cfg.BEHAVIOUR_MONTH_COL]).reset_index(drop=True)
    g = b[cfg.KEYS["customer"]]

    f = b[[cfg.KEYS["customer"], cfg.BEHAVIOUR_MONTH_COL]].copy()

    # C4 drivers first (memo §1.2) ------------------------------------------
    f["bounce_streak"]          = _consecutive_positive(b["nach_bounce_count"], g)
    f["bounce_count_3m"]        = _roll(b["nach_bounce_count"], g, W3, "sum")
    f["spending_shock_recency"] = _months_since_last(b["spending_shock_flag"], g)
    f["spending_shock_rate_3m"] = _roll(b["spending_shock_flag"], g, W3, "mean")
    f["utilisation_level"]      = b["credit_utilisation"]
    f["utilisation_creep_3m"]   = b["credit_utilisation"] - b.groupby(g)["credit_utilisation"].shift(W3)
    # balance volatility LEVEL — risk shows up as persistently high volatility, not
    # transient spikes. (Step-6: the z-vs-baseline spike divided by a near-zero
    # trailing std and exploded to +/-1e7; the level is stable and corr +0.55.)
    f["balance_volatility_level"] = b["balance_volatility"]
    f["cashflow_consistency_level"] = b["cashflow_consistency"]
    # slope over a 3-pt window == (y[t]-y[t-2])/2 exactly (OLS, evenly spaced)
    f["cashflow_consistency_slope"] = (b["cashflow_consistency"] - b.groupby(g)["cashflow_consistency"].shift(2)) / 2.0

    # rest of the panel -----------------------------------------------------
    f["inquiry_velocity_accel"] = b["bureau_inquiry_velocity"] - b.groupby(g)["bureau_inquiry_velocity"].shift(W3)
    f["app_engagement_drop"]    = b["app_engagement"] - b.groupby(g)["app_engagement"].shift(W3)
    f["first_bounce_flag"]      = _first_occurrence(b["nach_bounce_count"], g)

    return f


# --- repayment (loan-level) features ----------------------------------------

def repayment_features() -> pd.DataFrame:
    r = pd.read_parquet(cfg.PATHS["repayments"]).rename(columns={"period_index": "months_on_book"})
    r = r.sort_values([cfg.KEYS["loan"], "months_on_book"]).reset_index(drop=True)
    g = r[cfg.KEYS["loan"]]

    f = r[[cfg.KEYS["loan"], "months_on_book"]].copy()
    f["partial_run"]     = _consecutive_positive(r[cfg.REPAY_COLS["partial_flag"]], g)
    f["dpd_current"]     = r[cfg.REPAY_COLS["dpd"]]                 # level at t
    f["missed_cum_at_t"] = r[cfg.REPAY_COLS["missed_cum"]]         # level at t
    return f


# --- assemble feature matrix on the labelled spine --------------------------

FILL_LEVEL = {"utilisation_level", "cashflow_consistency_level", "balance_volatility_level"}
FEATURE_NAMES = [name for name, *_ in cfg.FEATURE_SPEC]
ESCALATION_NAMES = [name for name, *_ in cfg.ESCALATION_SPEC]   # carried, not modelled


def build_matrix() -> pd.DataFrame:
    spine = pd.read_parquet(cfg.OUT / "frame" / "labelled_spine.parquet")
    bf = behaviour_features()
    rf = repayment_features()

    # exact joins (each feature value is already <= its own month/MOB)
    m = spine.merge(
        bf, left_on=[cfg.KEYS["customer"], "obs_month"],
        right_on=[cfg.KEYS["customer"], cfg.BEHAVIOUR_MONTH_COL], how="left",
    ).drop(columns=[cfg.BEHAVIOUR_MONTH_COL])
    m = m.merge(rf, on=[cfg.KEYS["loan"], "months_on_book"], how="left")

    # fills: history-short windows -> 0 (no change/streak observed yet);
    #        level features -> median (neutral); recency already capped.
    for name in FEATURE_NAMES + ESCALATION_NAMES:
        if name in FILL_LEVEL:
            m[name] = m[name].fillna(m[name].median())
        else:
            m[name] = m[name].fillna(0)

    return m


# --- validation -------------------------------------------------------------

def _validate(m: pd.DataFrame) -> None:
    print("=" * 64)
    print("STEP 4 — POINT-IN-TIME FEATURES — validation")
    print("=" * 64)
    print(f"rows                : {len(m):,}")
    print(f"model features      : {len(FEATURE_NAMES)} (behaviour-led; matches FEATURE_SPEC: "
          f"{len(FEATURE_NAMES) == len(cfg.FEATURE_SPEC)})")
    print(f"escalation features : {len(ESCALATION_NAMES)} carried but NOT modelled "
          f"({', '.join(ESCALATION_NAMES)})")
    print(f"any NaN in features : {bool(m[FEATURE_NAMES + ESCALATION_NAMES].isna().any().any())}")
    print(f"_true_latent loaded : False  (never read in this module — A-019)")

    # Monotone-direction sanity: sign of each feature's correlation with label
    # should match the sign declared in FEATURE_SPEC (the GBM's monotone
    # constraint). A mismatch means the transform or the assumed direction is off.
    print("\nfeature vs label — does the empirical sign match FEATURE_SPEC?")
    print(f"  {'feature':<30}{'exp':>4}{'corr':>9}  match")
    bad = []
    for name, src, transform, sign in cfg.FEATURE_SPEC:
        c = m[name].corr(m["label"])
        emp = 0 if (pd.isna(c) or abs(c) < 1e-6) else (1 if c > 0 else -1)
        ok = (emp == sign) or emp == 0
        if not ok:
            bad.append(name)
        print(f"  {name:<30}{sign:>+4}{(0.0 if pd.isna(c) else c):>+9.3f}  "
              f"{'ok' if ok else 'MISMATCH'}")
    print(f"\n  sign mismatches: {bad if bad else 'none'}")

    # Done-when: a known defaulter's features deteriorate BEFORE the 90+ month.
    _spot_check_defaulter(m)


def _spot_check_defaulter(m: pd.DataFrame) -> None:
    rep = pd.read_parquet(cfg.PATHS["repayments"]).rename(columns={"period_index": "months_on_book"})
    D = rep.loc[rep.dpd >= cfg.DEFAULT_DPD].groupby(cfg.KEYS["loan"])["months_on_book"].min()
    # a defaulter with several pre-default seasoned months in the matrix
    cand = (m[m.label.notna()].groupby(cfg.KEYS["loan"]).size()
            .loc[lambda s: s >= 3].index)
    cand = [l for l in cand if l in D.index]
    if not cand:
        print("\n(no multi-month defaulter available for spot check)")
        return
    loan = cand[0]
    d = int(D.loc[loan])
    traj = (m[m[cfg.KEYS["loan"]] == loan]
            .sort_values("months_on_book")
            [["months_on_book", "bounce_streak", "cashflow_consistency_slope",
              "utilisation_level", "spending_shock_rate_3m", "label"]])
    print(f"\nspot check — defaulter {loan}, first 90+ at MOB {d} "
          f"(features should worsen as MOB approaches {d}; row at MOB {d} is absent by design):")
    print(traj.to_string(index=False))


if __name__ == "__main__":
    m = build_matrix()
    _validate(m)
    keep = ([cfg.KEYS["loan"], cfg.KEYS["customer"], "obs_month", "months_on_book",
             cfg.LOAN_COLS["product"], cfg.SEGMENT_COL] + FEATURE_NAMES
            + ESCALATION_NAMES + ["label"])
    m = m[keep]
    out = cfg.OUT / "frame" / "feature_matrix.parquet"
    m.to_parquet(out, index=False)
    m.head(200).to_csv(cfg.OUT / "frame" / "feature_matrix_sample.csv", index=False)
    print(f"\nwritten: {out}  ({len(m):,} rows x {len(FEATURE_NAMES)} features)")
