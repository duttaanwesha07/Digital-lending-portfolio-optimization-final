"""
frame/labels.py — Step 3: the forward-window 90+ label.

For each spine row at month t (a performing, seasoned loan-month), the target is:

    label = 1  if the loan first reaches 90+ DPD within months t+1 .. t+H
    label = 0  if it survives the window with no 90+  AND we observe that survival
    DROP       only if the loan is still Active at the data edge and the window
               runs off the end of observation  (genuinely unknown outcome)

The last line is the whole point of this step — but with one essential subtlety
that is easy to get wrong (and did, on the first pass): a loan that REACHES A
TERMINAL STATE before t+H (Closed/prepaid/matured, or Written-off) with no 90+
in the window is an OBSERVED SURVIVOR, label 0 — it cannot default after it
ends. Only a loan still ACTIVE at its last observed month is right-censored at
the snapshot, and only then, with an incomplete window, is the outcome unknown.

Coding completed loans as "censored" silently deletes the short-tenure BNPL
survivors and leaves only their defaulters behind — inflating those segments'
base rates to absurd levels. Coding a true data-edge censor as 0 fabricates a
good loan. Both are wrong; the loan_status split below avoids each.

LABELS MAY USE THE FUTURE. That is what a label is. The point-in-time rule
(config §5) constrains FEATURES, not the outcome. So we are free to use the
full outcome timeline here — summarised per loan as two facts:

    D = first_default_mob  : MOB of first 90+   (+inf if never)      -> the event
    L = last_observed_mob  : MOB the panel ends (snapshot/close/WO)  -> observability

Because each loan's panel is contiguous monthly (period_index == MOB), forward
windows in MOB-space equal calendar windows, so D and L are sufficient and exact.

Run:  python -m frame.labels
"""

import pandas as pd
import numpy as np

import config as cfg


def attach_labels(spine: pd.DataFrame | None = None) -> pd.DataFrame:
    if spine is None:
        spine = pd.read_parquet(cfg.OUT / "frame" / "observation_spine.parquet")

    rep = pd.read_parquet(cfg.PATHS["repayments"]).rename(
        columns={"period_index": "months_on_book"}
    )
    loans = pd.read_parquet(cfg.PATHS["loans"])
    loan_key, mob = cfg.KEYS["loan"], "months_on_book"
    status_col = cfg.LOAN_COLS["status"]
    H = cfg.LABEL_WINDOW_H

    # D: first month each loan reaches 90+ (the default event); +inf if never.
    D = (
        rep.loc[rep[cfg.REPAY_COLS["dpd"]] >= cfg.DEFAULT_DPD]
        .groupby(loan_key)[mob].min().rename("D")
    )
    # L: last month the loan is observed at all (panel end = snapshot/close/WO).
    L = rep.groupby(loan_key)[mob].max().rename("L")

    df = spine.merge(D, on=loan_key, how="left").merge(L, on=loan_key, how="left")
    df = df.merge(loans[[loan_key, status_col]], on=loan_key, how="left")
    df["D"] = df["D"].fillna(np.inf)
    t = df[mob]

    # Did the loan reach a TERMINAL state at L? (Closed/prepaid/matured or
    # Written-off.) If so, survival to L with no 90+ in the window is OBSERVED:
    # the loan cannot default after it ends. Only 'Active' loans are
    # right-censored at the data edge (the M24 snapshot).
    terminal = df[status_col] != "Active"

    # --- assign outcome ----------------------------------------------------
    defaults_in_window = df["D"] <= (t + H)                       # event seen in t+1..t+H
    observed_survivor = (df["D"] > (t + H)) & (terminal | (df["L"] >= (t + H)))
    censored = (df["D"] > (t + H)) & (~terminal) & (df["L"] < (t + H))  # Active + window off the edge

    df["label"] = np.where(defaults_in_window, 1,
                   np.where(observed_survivor, 0, -1))            # -1 = censored marker

    n_censored = int((df["label"] == -1).sum())
    labelled = df.loc[df["label"] != -1].drop(columns=["D", "L", status_col]).copy()
    labelled["label"] = labelled["label"].astype("int8")

    _report(spine, labelled, n_censored, H)
    return labelled


def _report(spine, lab, n_censored, H):
    seg, mob = cfg.SEGMENT_COL, "months_on_book"
    print("=" * 64)
    print(f"STEP 3 — FORWARD-WINDOW 90+ LABEL  (H = {H} months)")
    print("=" * 64)
    print(f"spine rows in                  : {len(spine):,}")
    print(f"dropped — censored window      : {n_censored:,}  "
          f"({n_censored/len(spine):.1%})  (unknown outcome, NOT coded 0)")
    print(f"labelled rows out              : {len(lab):,}")
    pos = int(lab.label.sum())
    print(f"  label = 1 (heads to 90+)     : {pos:,}")
    print(f"  label = 0 (survives window)  : {len(lab)-pos:,}")
    print(f"  base rate                    : {lab.label.mean():.2%}  (performing-month default-onset rate)")

    # DONE-WHEN checks
    ok_binary = set(lab.label.unique()).issubset({0, 1})
    ok_nonull = lab.label.notna().all()
    print("\nDONE-WHEN checks")
    print(f"  [{'PASS' if ok_binary else 'FAIL'}] every retained row is 0/1 "
          f"(values = {sorted(lab.label.unique())})")
    print(f"  [{'PASS' if ok_nonull else 'FAIL'}] no missing labels")
    plausible = 0.005 < lab.label.mean() < 0.15
    print(f"  [{'PASS' if plausible else 'CHECK'}] base rate is a few percent "
          f"({lab.label.mean():.2%})")

    print("\nbase rate by segment (expect Contain #3 & Subprime #6 highest)")
    by = lab.groupby(seg)["label"].agg(["mean", "size"]).sort_index()
    for s, row in by.iterrows():
        star = "  <- priority" if s in cfg.PRIORITY_SEGMENTS else ""
        print(f"  {s:<32} {row['mean']:>6.2%}   n={int(row['size']):>7,}{star}")

    print("\nlabelled rows & positives by observation month")
    bm = lab.groupby("obs_month")["label"].agg(["size", "sum"])
    for m, row in bm.iterrows():
        tail = "  <- OOT test" if m >= cfg.OOT_TEST_FROM_MONTH else ""
        if m > cfg.LAST_FULL_WINDOW_MONTH:
            tail = "  <- label=1 + completed survivors only (Active loans censored)"
        print(f"  {m}   n={int(row['size']):>7,}  pos={int(row['sum']):>5,}{tail}")


if __name__ == "__main__":
    labelled = attach_labels()
    out = cfg.OUT / "frame" / "labelled_spine.parquet"
    labelled.to_parquet(out, index=False)
    labelled.head(200).to_csv(cfg.OUT / "frame" / "labelled_spine_sample.csv", index=False)
    print(f"\nwritten: {out}  ({len(labelled):,} rows)")
