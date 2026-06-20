"""
validate/validation.py — Step 7: the out-of-time validation.

Proves the flag (a) discriminates, (b) fires EARLY, (c) generalises, (d) isn't
leaking. The headline is not AUC — it is LEAD TIME. A model with great AUC and
no lead time has failed F's job.

Evaluation population matters. The EWS serves loans NOT already escalated, so we
report three views and lead with the honest one:
  - clean        : dpd_current == 0   (zero surface delinquency — strictest early test)
  - pre-escalation: dpd_current < 30  (the EWS's operating book; >=30 -> collections)
  - full performing: all OOT-test rows (CONTEXT ONLY — inflated by already-late loans)

Primary model = scorecard (governable). GBM reported alongside.

Run:  python -m validate.validation
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

import config as cfg

DAYS = 30.44
PRIMARY = "score_scorecard"


def _load():
    sc = pd.read_parquet(cfg.OUT / "scores" / "scores.parquet")
    fm = pd.read_parquet(cfg.OUT / "frame" / "feature_matrix_split.parquet")
    sc = sc.merge(fm[[cfg.KEYS["loan"], "months_on_book", "dpd_current"]],
                  on=[cfg.KEYS["loan"], "months_on_book"], how="left")
    sc["obs_p"] = pd.PeriodIndex(sc["obs_month"], freq="M")
    return sc


def _ks(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.max(tpr - fpr))


def _decile_lift(df, score):
    d = df.assign(dec=pd.qcut(df[score].rank(method="first"), 10, labels=False))
    base = df.label.mean()
    top = d[d.dec == 9].label.mean()
    return top / base if base > 0 else np.nan


def _pr_table(df, score, fracs=(0.05, 0.10, 0.20)):
    rows = []
    for fr in fracs:
        thr = df[score].quantile(1 - fr)
        flagged = df[score] >= thr
        tp = int((flagged & (df.label == 1)).sum())
        prec = tp / max(int(flagged.sum()), 1)
        rec = tp / max(int((df.label == 1).sum()), 1)
        rows.append((fr, thr, int(flagged.sum()), prec, rec))
    return rows


def _lead_time(work, test_def, D_month, score, thr, pre_escalation_only=True):
    """Days between first EARLY flag and the 90+ event, for OOT-period defaulters."""
    w = work[work[cfg.KEYS["loan"]].isin(test_def)].copy()
    if pre_escalation_only:
        w = w[w["dpd_current"] < 30]            # the flag must fire while not-yet-escalated
    w = w[w[score] >= thr]
    first = w.sort_values("obs_p").groupby(cfg.KEYS["loan"]).first()
    lead_m = (first.index.map(D_month).to_series().values
              - first["obs_p"].apply(lambda p: p.ordinal).values)
    lead_days = lead_m * DAYS
    flagged_early = len(first)
    return {
        "n_test_defaulters": len(test_def),
        "flagged_early": flagged_early,
        "early_detection_rate": flagged_early / max(len(test_def), 1),
        "mean_lead_days": float(np.mean(lead_days)) if flagged_early else 0.0,
        "median_lead_days": float(np.median(lead_days)) if flagged_early else 0.0,
    }


def run():
    sc = _load()
    te = sc[sc.split_oot == "test"].copy()

    pops = {
        "clean (dpd=0)":        te[te.dpd_current == 0],
        "pre-escalation (<30)": te[te.dpd_current < 30],
        "full performing":      te,
    }

    print("=" * 70); print("STEP 7 — OUT-OF-TIME VALIDATION  (primary = scorecard)"); print("=" * 70)
    print("\nDISCRIMINATION by evaluation population")
    print(f"  {'population':<22}{'n':>8}{'pos':>6}{'base':>7}{'AUC':>8}{'KS':>7}{'top-decile lift':>17}")
    for name, d in pops.items():
        if d.label.nunique() < 2:
            continue
        auc = roc_auc_score(d.label, d[PRIMARY]); ks = _ks(d.label, d[PRIMARY])
        lift = _decile_lift(d, PRIMARY)
        tag = "  <- HONEST early-warning" if name.startswith("clean") else (
              "  <- operating book" if name.startswith("pre") else "  <- CONTEXT (inflated)")
        print(f"  {name:<22}{len(d):>8,}{int(d.label.sum()):>6}{d.label.mean():>6.2%}"
              f"{auc:>8.3f}{ks:>7.3f}{lift:>13.1f}x   {tag}")

    # ---- precision / recall on the operating book ----
    op = pops["pre-escalation (<30)"]
    print("\nPRECISION / RECALL — operating book (pre-escalation), scorecard")
    print(f"  {'flag top':>9}{'threshold':>11}{'#flagged':>10}{'precision':>11}{'recall':>9}")
    for fr, thr, nfl, prec, rec in _pr_table(op, PRIMARY):
        print(f"  {fr:>8.0%}{thr:>11.3f}{nfl:>10,}{prec:>10.1%}{rec:>9.1%}")

    # ---- LEAD TIME (the headline) — OUT-OF-SAMPLE only ----
    rep = pd.read_parquet(cfg.PATHS["repayments"])
    Dcal = rep[rep.dpd >= cfg.DEFAULT_DPD].groupby(cfg.KEYS["loan"])["period_date"].min()
    D_ord = pd.Series(pd.PeriodIndex(Dcal, freq="M").map(lambda p: p.ordinal), index=Dcal.index)
    cut = pd.Period(cfg.OOT_TEST_FROM_MONTH, freq="M")
    D_ord_map = D_ord.to_dict()
    test_def = set(D_ord[D_ord >= cut.ordinal].index)        # defaults occurring in the OOT period

    # OOS scoring opportunities: test-window, pre-escalation rows of those defaulters.
    # Only loans with >=1 such row had a fair OOS chance to be flagged early.
    te["obs_ord"] = te["obs_p"].apply(lambda p: p.ordinal)
    oos = te[te[cfg.KEYS["loan"]].isin(test_def) & (te.dpd_current < 30)].copy()
    eligible = oos[cfg.KEYS["loan"]].unique()

    print("\nLEAD TIME — OUT-OF-SAMPLE (flag searched in test window only, while <30 DPD)")
    print(f"  OOT-period defaulters with an OOS early-warning opportunity: {len(eligible):,}")
    print(f"  {'flag top':>9}{'thr':>9}{'early-detected':>16}{'mean lead':>11}{'median':>9}")
    op_for_thr = pops["pre-escalation (<30)"][PRIMARY]
    lead_headline = 0.0
    for fr in (0.05, 0.10, 0.20):
        thr = op_for_thr.quantile(1 - fr)
        w = oos[oos[PRIMARY] >= thr]
        first = w.sort_values("obs_ord").groupby(cfg.KEYS["loan"]).first()
        if len(first):
            lead_m = np.array([D_ord_map[l] for l in first.index]) - first["obs_ord"].values
            ld = lead_m * DAYS
            if abs(fr - 0.10) < 1e-9:
                lead_headline = float(np.mean(ld))
            print(f"  {fr:>8.0%}{thr:>9.3f}{len(first)/len(eligible):>15.0%}"
                  f"{np.mean(ld):>10.0f}d{np.median(ld):>7.0f}d")
        else:
            print(f"  {fr:>8.0%}{thr:>9.3f}{'0%':>16}{'-':>11}{'-':>9}")
    print("  (OOS lead is capped by the 3-month test window; the population lead curve")
    print("   below shows behavioural deterioration begins ~6 months out.)")

    # ---- leakage / point-in-time audit ----
    print("\nLEAKAGE / POINT-IN-TIME AUDIT")
    lat = pd.read_parquet(cfg.PATHS["true_latent"])
    aud = te.merge(lat, on=cfg.KEYS["customer"], how="left")
    corr = aud[PRIMARY].corr(aud["true_latent_risk"])
    print(f"  corr(score, true_latent_risk) = {corr:+.3f}  "
          f"(expect NEGATIVE: higher creditworthiness -> lower flag; sanity {'OK' if corr < 0 else 'CHECK'})")
    print(f"  true_latent_risk in model features? {('true_latent_risk' in [n for n,*_ in cfg.FEATURE_SPEC])}  "
          f"(must be False — A-019)")
    # features lead, not coincide: score rises as default approaches
    md = te[te[cfg.KEYS["loan"]].isin(test_def)].copy()
    md["mtd"] = md[cfg.KEYS["loan"]].map(D_ord_map) - md["obs_ord"]
    by = md[md.mtd.between(1, 6)].groupby("mtd")[PRIMARY].mean()
    print(f"  mean score by months-to-default (6->1, should RISE): "
          f"{', '.join(f'{m}:{v:.2f}' for m, v in by.items())}")

    # ---- stability ----
    print("\nSTABILITY")
    print("  AUC by priority segment (operating book):")
    for s in cfg.PRIORITY_SEGMENTS:
        d = op[op.segment == s]
        if d.label.nunique() == 2:
            print(f"    {s:<30} AUC {roc_auc_score(d.label, d[PRIMARY]):.3f}  (n={len(d):,}, pos={int(d.label.sum())})")
    rng = cfg.get_rng("validate", "bootstrap")
    aucs = []
    arr_y = op.label.values; arr_s = op[PRIMARY].values; n = len(op)
    for _ in range(cfg.BOOTSTRAP_RESAMPLES):
        idx = rng.integers(0, n, n)
        if arr_y[idx].sum() > 0:
            aucs.append(roc_auc_score(arr_y[idx], arr_s[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    print(f"  bootstrap AUC 95% CI (operating book, {cfg.BOOTSTRAP_RESAMPLES}x): "
          f"[{lo:.3f}, {hi:.3f}]  mean {np.mean(aucs):.3f}")

    # ---- self-cure context ----
    print(f"\n  NOTE: ~{cfg.SELF_CURE_EPISODE_RATE:.0%} of stress episodes self-resolve (A-046); "
          f"some amber flags WILL cure. This caps achievable precision — by design, not a defect.")

    # ---- Gate 4 verdict ----
    clean = pops["clean (dpd=0)"]
    auc_h = roc_auc_score(clean.label, clean[PRIMARY]); ks_h = _ks(clean.label, clean[PRIMARY])
    lead_days = lead_headline      # OOS mean lead at top-10% flag (computed above)
    g = cfg.GATE4
    checks = [
        ("lead_time", lead_days >= g["lead_time_days_min"], f"{lead_days:.0f}d >= {g['lead_time_days_min']}d"),
        ("oot_auc",   auc_h >= g["oot_auc_min"],            f"{auc_h:.3f} >= {g['oot_auc_min']}"),
        ("oot_ks",    ks_h >= g["oot_ks_min"],              f"{ks_h:.3f} >= {g['oot_ks_min']}"),
        ("lift",      _decile_lift(clean, PRIMARY) >= g["top_decile_lift_min"],
                      f"{_decile_lift(clean, PRIMARY):.1f}x >= {g['top_decile_lift_min']}x"),
        ("no_leakage", corr < 0 and "true_latent_risk" not in [n for n,*_ in cfg.FEATURE_SPEC], "audit clean"),
        ("stable",    (hi - lo) < 0.15,                     f"CI width {hi-lo:.3f} < 0.15"),
    ]
    print("\n" + "=" * 70); print("GATE 4 — EWS REVIEW"); print("=" * 70)
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<12} {detail}")
    verdict = "PASS" if all(ok for _, ok, _ in checks) else "FAIL — fix before sign-off"
    print(f"\n  GATE 4 VERDICT: {verdict}")

    out = cfg.OUT_DIRS["validation"]; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"check": n, "pass": ok, "detail": d} for n, ok, d in checks]).to_csv(
        out / "gate4_results.csv", index=False)
    print(f"\nwritten: {out/'gate4_results.csv'}")


if __name__ == "__main__":
    run()
