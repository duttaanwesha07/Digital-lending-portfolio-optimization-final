"""
model/split.py — Step 5: the train/test splits.

Two splits, two jobs.

WITHIN-TIME  (tuning) — loans-whole, random, stratified by segment x has-positive.
  A loan is wholly in train or wholly in test. Cheap and correct here; used only
  to tune hyper-parameters. The within-time number always flatters; it is NOT the
  headline.

OUT-OF-TIME  (the real proof) — PURGED + EMBARGOED temporal split.
  Why not a strict loans-whole observation cut? On this panel that discards 71%
  of pre-cut rows (long-lived loans straddle the boundary), leaving train smaller
  than test — a broken test. The cohort-youngest alternative yields ~3k age-skewed
  rows. So we use the standard credit-risk OOT instead:

    train   : obs_month whose FULL forward window closes before the cut
              (obs_month <= cut - (H+1))   -> label cannot see the test period
    embargo : the H months just before the cut are DROPPED from both sides
              (cut-H .. cut-1)             -> purges label leakage at the seam
    test    : the recent FULL-window quarter (cut .. LAST_FULL_WINDOW_MONTH)
    excluded: truncated-window months after LAST_FULL_WINDOW_MONTH (biased; out)

  A long-lived loan may appear in train (early months) and test (recent months).
  With strictly point-in-time features (Step 4) and the embargo gap, NO outcome
  information crosses the boundary — this is the production reality (you know a
  customer's history when you score them later), not leakage. *** This is a
  deliberate, flagged deviation from the manual's "loans-whole for every split";
  it needs reviewer ratification. ***

Run:  python -m model.split
"""

import numpy as np
import pandas as pd

import config as cfg


def _within_time(m: pd.DataFrame) -> pd.Series:
    """Loans-whole random split, stratified by (segment, has-positive)."""
    loan = (m.groupby(cfg.KEYS["loan"])
              .agg(segment=(cfg.SEGMENT_COL, "first"),
                   any_pos=("label", "max")).reset_index())
    loan["stratum"] = loan["segment"].astype(str) + "|" + loan["any_pos"].astype(str)

    rng = cfg.get_rng("split", "within_time")
    test_loans = set()
    for _, grp in loan.groupby("stratum"):
        ids = grp[cfg.KEYS["loan"]].to_numpy()
        rng.shuffle(ids)
        n_test = int(round(len(ids) * cfg.WITHIN_TIME_TEST_FRAC))
        test_loans.update(ids[:n_test].tolist())

    return np.where(m[cfg.KEYS["loan"]].isin(test_loans), "test", "train")


def _out_of_time(m: pd.DataFrame) -> pd.Series:
    H = cfg.LABEL_WINDOW_H
    p = pd.PeriodIndex(m["obs_month"], freq="M")
    cut = pd.Period(cfg.OOT_TEST_FROM_MONTH, freq="M")
    last_full = pd.Period(cfg.LAST_FULL_WINDOW_MONTH, freq="M")
    train_max = cut - (H + 1)            # full forward window closes before cut

    out = np.full(len(m), "excluded", dtype=object)
    out[p <= train_max] = "train"
    out[(p > train_max) & (p < cut)] = "embargo"      # purged seam
    out[(p >= cut) & (p <= last_full)] = "test"
    # p > last_full stays "excluded" (truncated, biased windows)
    return out


def build_splits() -> pd.DataFrame:
    m = pd.read_parquet(cfg.OUT / "frame" / "feature_matrix.parquet")
    m["split_within"] = _within_time(m)
    m["split_oot"] = _out_of_time(m)
    _validate(m)
    return m


def _seg_rate(df):
    return f"{len(df):>7,} rows  pos={int(df.label.sum()):>5,} ({df.label.mean():.2%})"


def _validate(m: pd.DataFrame) -> None:
    lk = cfg.KEYS["loan"]
    print("=" * 64)
    print("STEP 5 — SPLITS — validation")
    print("=" * 64)

    # ---- within-time ----
    wt_tr = m[m.split_within == "train"]; wt_te = m[m.split_within == "test"]
    tr_loans = set(wt_tr[lk]); te_loans = set(wt_te[lk])
    print("\nWITHIN-TIME (loans-whole, stratified) — tuning only")
    print(f"  train : {_seg_rate(wt_tr)}")
    print(f"  test  : {_seg_rate(wt_te)}")
    overlap_wt = len(tr_loans & te_loans)
    print(f"  [{'PASS' if overlap_wt==0 else 'FAIL'}] no loan on both sides "
          f"(overlap={overlap_wt})")
    print(f"  base-rate train vs test: {wt_tr.label.mean():.2%} vs {wt_te.label.mean():.2%} "
          f"(stratification {'holds' if abs(wt_tr.label.mean()-wt_te.label.mean())<0.005 else 'CHECK'})")

    # ---- out-of-time ----
    print("\nOUT-OF-TIME (purged + embargoed temporal)")
    for part in ["train", "embargo", "test", "excluded"]:
        d = m[m.split_oot == part]
        mn = d.obs_month.min() if len(d) else "-"; mx = d.obs_month.max() if len(d) else "-"
        tag = {"train": "  -> fit", "embargo": "  -> dropped (purge seam)",
               "test": "  -> THE headline test", "excluded": "  -> truncated, out"}[part]
        print(f"  {part:<9} {_seg_rate(d)}   months {mn}..{mx}{tag}")

    oot_tr = m[m.split_oot == "train"]; oot_te = m[m.split_oot == "test"]
    # done-when: test strictly later than train; embargo separates them
    ok_order = oot_tr.obs_month.max() < oot_te.obs_month.min()
    print(f"\n  [{'PASS' if ok_order else 'FAIL'}] test months strictly later than train "
          f"({oot_tr.obs_month.max()} < {oot_te.obs_month.min()})")
    both = len(set(oot_tr[lk]) & set(oot_te[lk]))
    print(f"  [INFO] loans in BOTH train & test : {both:,}  "
          f"(allowed & production-faithful — features point-in-time + {cfg.LABEL_WINDOW_H}-mo embargo; "
          f"NO label crosses the seam)")
    # the leakage that WOULD matter is purged: confirm no train label reaches cut
    cut = pd.Period(cfg.OOT_TEST_FROM_MONTH, freq="M")
    train_label_end = (pd.PeriodIndex(oot_tr.obs_month, freq="M") + cfg.LABEL_WINDOW_H).max()
    print(f"  [{'PASS' if train_label_end < cut else 'FAIL'}] latest train label window "
          f"ends {train_label_end} < cut {cut}  (no forward-label leakage across seam)")


if __name__ == "__main__":
    (cfg.OUT / "frame").mkdir(parents=True, exist_ok=True)
    m = build_splits()
    out = cfg.OUT / "frame" / "feature_matrix_split.parquet"
    m.to_parquet(out, index=False)
    print(f"\nwritten: {out}")
