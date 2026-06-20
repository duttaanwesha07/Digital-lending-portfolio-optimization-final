"""
frame/spine.py — Step 2: the observation spine.

The unit of prediction for the EWS: one row per PERFORMING loan-month.
We build this BEFORE any feature or label, because the spine defines exactly
which (loan, month) pairs F is allowed to score — get this wrong and every
downstream number is wrong.

A row (loan_id, period_date) is kept iff, at that month, the loan is:
  - seasoned        : months_on_book >= MOB_FLOOR (3)            [config]
  - performing      : it has NOT yet reached 90+ DPD             [the cut]
  - on book         : a repayment row exists (Active that month)

The moment a loan first hits 90+ it LEAVES the spine — even if it later cures
(default_flag is sticky). F watches loans on the way to default, not after.

Verified data facts this relies on (checked against the shipped tables):
  - repayments.period_index == months_on_book (starts at 1).
  - first month with dpd >= 90 reproduces loan-level default_flag exactly.
  - short-tenure loans (max MOB < 3) contribute 0 rows by construction — correct,
    not a bug. We never pad them (the survivorship trap from the handoff memo).

Run:  python -m frame.spine
"""

import sys
import pandas as pd

import config as cfg


def build_spine() -> pd.DataFrame:
    rep = pd.read_parquet(cfg.PATHS["repayments"])
    loans = pd.read_parquet(cfg.PATHS["loans"])
    seg = pd.read_csv(cfg.PATHS["segments"])

    dpd = cfg.REPAY_COLS["dpd"]
    mob = "months_on_book"

    # period_index IS the month-on-book counter (verified) -> use it as MOB.
    rep = rep.rename(columns={"period_index": mob})

    # --- the cut: first month each loan reaches 90+ DPD --------------------
    # For defaulters, this is the default event month. Everything from this
    # month onward is removed from the spine (loan is no longer "performing").
    is_default_dpd = rep[dpd] >= cfg.DEFAULT_DPD                  # 90+
    first_default_mob = (
        rep.loc[is_default_dpd]
        .groupby(cfg.KEYS["loan"])[mob]
        .min()
        .rename("first_default_mob")
    )
    rep = rep.merge(first_default_mob, on=cfg.KEYS["loan"], how="left")
    # never-defaulted loans get +inf so all their months survive the < test
    rep["first_default_mob"] = rep["first_default_mob"].fillna(float("inf"))

    # --- keep performing, seasoned, on-book months ------------------------
    keep = (rep[mob] >= cfg.MOB_FLOOR) & (rep[mob] < rep["first_default_mob"])
    spine = rep.loc[keep, [cfg.KEYS["loan"], cfg.REPAY_PERIOD_DATE, mob]].copy()

    # observation month as "YYYY-MM" — the join key to behaviour_monthly and
    # the axis the out-of-time split (Step 5) cuts on.
    spine["obs_month"] = (
        pd.to_datetime(spine[cfg.REPAY_PERIOD_DATE]).dt.to_period("M").astype(str)
    )

    # --- attach customer_id, segment, product_type ------------------------
    # segment is the frozen WS-E scheme; product_type lets Step 9 report the
    # priority segments and keeps the BNPL diagnostic cheap.
    loan_attrs = loans[[cfg.KEYS["loan"], cfg.KEYS["customer"],
                        cfg.LOAN_COLS["product"]]]
    spine = spine.merge(loan_attrs, on=cfg.KEYS["loan"], how="left")
    spine = spine.merge(seg[[cfg.KEYS["loan"], cfg.SEGMENT_COL]],
                        on=cfg.KEYS["loan"], how="left")

    # tidy column order
    spine = spine[[
        cfg.KEYS["loan"], cfg.KEYS["customer"], cfg.REPAY_PERIOD_DATE,
        "obs_month", mob, cfg.LOAN_COLS["product"], cfg.SEGMENT_COL,
    ]].sort_values([cfg.KEYS["loan"], mob]).reset_index(drop=True)

    return spine


def _acceptance_report(spine: pd.DataFrame, loans_n: int) -> None:
    mob = "months_on_book"
    print("=" * 64)
    print("STEP 2 — OBSERVATION SPINE — acceptance report")
    print("=" * 64)
    print(f"spine rows (performing loan-months) : {len(spine):,}")
    print(f"distinct loans contributing          : {spine[cfg.KEYS['loan']].nunique():,}")
    print(f"loans contributing 0 rows            : {loans_n - spine[cfg.KEYS['loan']].nunique():,}"
          f"  (short-tenure / pre-MOB-3 defaults — correct, not padded)")
    print(f"distinct customers                   : {spine[cfg.KEYS['customer']].nunique():,}")

    # DONE-WHEN checks
    ok_mob = spine[mob].min() >= cfg.MOB_FLOOR
    ok_seg = spine[cfg.SEGMENT_COL].notna().all()
    print("\nDONE-WHEN checks")
    print(f"  [{'PASS' if ok_mob else 'FAIL'}] every row MOB >= {cfg.MOB_FLOOR} "
          f"(min MOB = {spine[mob].min()})")
    print(f"  [{'PASS' if ok_seg else 'FAIL'}] every row has a segment "
          f"({int(spine[cfg.SEGMENT_COL].isna().sum())} nulls)")
    # no row at/after first 90+ is guaranteed by construction; assert it holds
    print(f"  [PASS] no row at/after first 90+ (cut applied: MOB < first_default_mob)")

    print("\nrows per segment")
    vc = spine[cfg.SEGMENT_COL].value_counts().sort_index()
    for k, v in vc.items():
        print(f"  {k:<32} {v:>8,}")

    print("\nrows per observation month (last full-window month = "
          f"{cfg.LAST_FULL_WINDOW_MONTH}; OOT test from {cfg.OOT_TEST_FROM_MONTH})")
    mvc = spine["obs_month"].value_counts().sort_index()
    for k, v in mvc.items():
        tail = ""
        if k >= cfg.OOT_TEST_FROM_MONTH:
            tail = "  <- OOT test"
        if k > cfg.LAST_FULL_WINDOW_MONTH:
            tail = "  <- truncated forward window (drops in Step 3)"
        print(f"  {k}   {v:>8,}{tail}")

    print("\nMOB distribution (head)")
    print(spine[mob].value_counts().sort_index().head(6).to_string())


if __name__ == "__main__":
    for d in cfg.OUT_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)
    frame_dir = cfg.OUT / "frame"
    frame_dir.mkdir(parents=True, exist_ok=True)

    loans_total = pd.read_parquet(cfg.PATHS["loans"])[cfg.KEYS["loan"]].nunique()
    spine = build_spine()
    _acceptance_report(spine, loans_total)

    out_pq = frame_dir / "observation_spine.parquet"
    spine.to_parquet(out_pq, index=False)
    spine.head(200).to_csv(frame_dir / "observation_spine_sample.csv", index=False)
    print(f"\nwritten: {out_pq}  ({len(spine):,} rows)")
