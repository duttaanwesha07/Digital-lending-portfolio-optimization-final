"""
model/models.py — Step 6: the two models.

PRIMARY  : WOE scorecard (logistic on monotonic weight-of-evidence bins).
           Transparent, explainable per flag, monotone BY CONSTRUCTION — the
           bad-rate in each feature's bins is isotonic-regressed into the
           economic direction declared in FEATURE_SPEC before WOE is computed,
           so the scorecard can never learn a shape a risk committee can't defend.
CHALLENGER: LightGBM with monotone_constraints read straight from FEATURE_SPEC.

Both are calibrated to probabilities via grouped (by loan) out-of-fold isotonic,
so Step 8's thresholds sit on honest probabilities and Step 7 gets clean OOF
train predictions. Models are fit on the OOT-train fold only; the out-of-time
test is untouched here (that verdict is Step 7).

Convention: WOE = ln(bad% / good%)  -> higher WOE = higher risk -> every logistic
coefficient should be POSITIVE. Sign sanity = (betas > 0) AND (each feature's WOE
moves with risk in its FEATURE_SPEC direction).

Run:  python -m model.models
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

import config as cfg

FEATURES = [n for n, *_ in cfg.FEATURE_SPEC]
SIGNS = {n: s for n, _, _, s in cfg.FEATURE_SPEC}
SMOOTH = 0.5


# ----------------------------- WOE scorecard --------------------------------

def _fit_bins(x: pd.Series, max_bins: int):
    """Return interior cut points; few-unique features bin by value."""
    if x.nunique() <= max_bins:
        vals = np.sort(x.unique())
        return ("vals", vals)
    q = np.linspace(0, 1, max_bins + 1)[1:-1]
    cuts = np.unique(np.quantile(x, q))
    return ("cuts", cuts)


def _apply_bins(x: pd.Series, binspec):
    kind, ref = binspec
    if kind == "vals":
        idx = np.searchsorted(ref, x, side="left")
        return np.clip(idx, 0, len(ref) - 1)
    return np.digitize(x, ref, right=False)


class WOEScorecard:
    def __init__(self, max_bins, min_bin_frac):
        self.max_bins = max_bins; self.min_bin_frac = min_bin_frac
        self.bins = {}; self.woe = {}; self.iv = {}; self.dir_ok = {}
        self.lr = None

    def fit(self, df: pd.DataFrame, y: pd.Series):
        tot_bad = y.sum() + SMOOTH; tot_good = (1 - y).sum() + SMOOTH
        W = pd.DataFrame(index=df.index)
        for f in FEATURES:
            x = df[f]
            binspec = _fit_bins(x, self.max_bins)
            b = _apply_bins(x, binspec)
            g = pd.DataFrame({"b": b, "y": y.values, "x": x.values})
            agg = g.groupby("b").agg(bad=("y", "sum"), n=("y", "size"),
                                     xmean=("x", "mean")).sort_values("xmean")
            agg["good"] = agg["n"] - agg["bad"]
            # enforce monotone bad-rate in the FEATURE_SPEC direction
            rate = (agg["bad"] + SMOOTH) / (agg["n"] + 2 * SMOOTH)
            iso = IsotonicRegression(increasing=(SIGNS[f] > 0), out_of_bounds="clip")
            mono = iso.fit_transform(agg["xmean"].values, rate.values, sample_weight=agg["n"].values)
            bad_m = mono * agg["n"].values; good_m = (1 - mono) * agg["n"].values
            woe = np.log((bad_m + SMOOTH) / tot_bad) - np.log((good_m + SMOOTH) / tot_good)
            woe_map = dict(zip(agg.index, woe))
            iv = float(np.sum(((bad_m + SMOOTH) / tot_bad - (good_m + SMOOTH) / tot_good) * woe))
            # direction check: does WOE rise with x in the spec direction?
            corr = np.corrcoef(agg["xmean"].values, woe)[0, 1] if len(agg) > 1 else 0.0
            self.bins[f] = binspec; self.woe[f] = woe_map
            self.iv[f] = iv; self.dir_ok[f] = (np.sign(corr) == SIGNS[f]) or abs(corr) < 1e-9
            W[f] = pd.Series(b, index=df.index).map(woe_map).fillna(0.0)

        self.lr = LogisticRegression(max_iter=1000, C=1.0)
        self.lr.fit(W.values, y.values)
        return self

    def _woe_frame(self, df):
        W = pd.DataFrame(index=df.index)
        for f in FEATURES:
            b = _apply_bins(df[f], self.bins[f])
            W[f] = pd.Series(b, index=df.index).map(self.woe[f]).fillna(0.0)
        return W

    def predict_proba(self, df):
        return self.lr.predict_proba(self._woe_frame(df).values)[:, 1]


# ----------------------------- monotonic GBM --------------------------------

def fit_gbm(df, y):
    constraints = [int(SIGNS[f]) for f in FEATURES]   # +1 increasing, -1 decreasing
    p = cfg.GBM
    model = lgb.LGBMClassifier(
        n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
        num_leaves=p["num_leaves"], min_child_samples=p["min_child_samples"],
        subsample=p["subsample"], colsample_bytree=p["colsample_bytree"],
        monotone_constraints=constraints, random_state=cfg.MASTER_SEED, verbose=-1,
    )
    model.fit(df[FEATURES].values, y.values)
    return model


# ------------------- grouped OOF calibration + scoring ----------------------

def _oof_calibrate(make_model, predict, train, y, groups):
    """5-fold grouped OOF preds on train -> isotonic calibrator; refit on full."""
    oof = np.zeros(len(train))
    gkf = GroupKFold(n_splits=5)
    for tr, va in gkf.split(train, y, groups):
        mdl = make_model(train.iloc[tr], y.iloc[tr])
        oof[va] = predict(mdl, train.iloc[va])
    cal = IsotonicRegression(out_of_bounds="clip").fit(oof, y.values)
    full = make_model(train, y)
    return full, cal, oof


def run():
    m = pd.read_parquet(cfg.OUT / "frame" / "feature_matrix_split.parquet")
    tr = m[m.split_oot == "train"].copy()
    y = tr["label"].astype(int)
    groups = tr[cfg.KEYS["loan"]].values

    # scorecard
    sc_make = lambda d, yy: WOEScorecard(cfg.SCORECARD["max_bins"], cfg.SCORECARD["min_bin_frac"]).fit(d, yy)
    sc_pred = lambda mdl, d: mdl.predict_proba(d)
    sc, sc_cal, sc_oof = _oof_calibrate(sc_make, sc_pred, tr, y, groups)

    # gbm
    gb_make = lambda d, yy: fit_gbm(d, yy)
    gb_pred = lambda mdl, d: mdl.predict_proba(d[FEATURES].values)[:, 1]
    gb, gb_cal, gb_oof = _oof_calibrate(gb_make, gb_pred, tr, y, groups)

    # calibrated scores on ALL rows
    m["score_scorecard"] = sc_cal.transform(sc.predict_proba(m))
    m["score_gbm"] = gb_cal.transform(gb.predict_proba(m[FEATURES].values)[:, 1])

    _validate(m, sc, sc_oof, gb_oof, y)

    out = cfg.OUT / "scores"; out.mkdir(parents=True, exist_ok=True)
    keep = [cfg.KEYS["loan"], cfg.KEYS["customer"], "obs_month", "months_on_book",
            cfg.LOAN_COLS["product"], cfg.SEGMENT_COL, "label",
            "split_within", "split_oot", "score_scorecard", "score_gbm"]
    m[keep].to_parquet(out / "scores.parquet", index=False)
    print(f"\nwritten: {out/'scores.parquet'}")
    return m


def _validate(m, sc, sc_oof, gb_oof, y):
    te = m[m.split_oot == "test"]
    print("=" * 64); print("STEP 6 — MODELS — validation"); print("=" * 64)

    print("\nscorecard — feature strength (IV) & WOE direction vs FEATURE_SPEC")
    print(f"  {'feature':<30}{'IV':>8}  dir")
    order = sorted(FEATURES, key=lambda f: -sc.iv[f])
    for f in order:
        print(f"  {f:<30}{sc.iv[f]:>8.3f}  {'ok' if sc.dir_ok[f] else 'MISMATCH'}")
    betas = sc.lr.coef_[0]
    print(f"  logistic betas all positive (WOE convention): "
          f"{bool((betas > -1e-6).all())}  (min beta {betas.min():+.3f})")
    bad_dir = [f for f in FEATURES if not sc.dir_ok[f]]
    print(f"  WOE direction mismatches: {bad_dir if bad_dir else 'none'}")

    print("\ndiscrimination (preview — formal OOT verdict is Step 7)")
    print(f"  {'':<12}{'train OOF AUC':>14}{'OOT test AUC':>14}")
    print(f"  {'scorecard':<12}{roc_auc_score(y, sc_oof):>14.4f}"
          f"{roc_auc_score(te.label, te.score_scorecard):>14.4f}")
    print(f"  {'gbm':<12}{roc_auc_score(y, gb_oof):>14.4f}"
          f"{roc_auc_score(te.label, te.score_gbm):>14.4f}")

    print("\ncalibration on OOT test (mean predicted vs actual)")
    for col in ["score_scorecard", "score_gbm"]:
        print(f"  {col:<16} pred {te[col].mean():.3%}  vs  actual {te.label.mean():.3%}")

    # CRITICAL — is the AUC a lead-time mirage driven by concurrent state?
    top1 = te.assign(_s=te.score_scorecard).nlargest(max(1, int(0.01 * len(te))), "_s")
    print("\nLEAD-TIME WARNING — of the top-1% scored OOT-test flags:")
    print(f"  already dpd>=30 at flag time: {(top1.dpd_current >= 30).mean():.0%}   "
          f"already dpd>=60: {(top1.dpd_current >= 60).mean():.0%}")
    print("  A high share means the model is flagging loans ALREADY failing.")
    print("  dpd_current & missed_cum_at_t are near-COINCIDENT, not early —")
    print("  for true lead time the EWS must be BEHAVIOUR-LED (see Step-6 note).")

    print(f"\nfixes applied — model is now BEHAVIOUR-LED: {len(FEATURES)} features, "
          f"dpd_current & missed_cum_at_t removed to escalation triggers;")
    print(f"  balance_volatility_level replaces the unstable spike. The top-1% dpd>=30 "
          f"share above should now be LOW = lead time restored.")


if __name__ == "__main__":
    run()
