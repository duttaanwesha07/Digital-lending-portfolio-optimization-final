"""
policy/segment_overlay.py — Step 9: segment overlay + watchlist.

Connects F back to E and hands a usable list downstream.
  1. Flag rates by segment, priority on Contain (#3) and Subprime BNPL (#6).
  2. The behavioural-distress cohort (E's ~66%-default "C4" pattern, defined by
     bounces / shocks / utilisation) surfaced at the TOP of the watchlist — now a
     live, leakage-clean score rather than a coincident label.
  3. The current-month watchlist (score + RAG band + segment) for:
        G  - prices / limits off the score
        I  - the policy lever "route Contain to early-warning monitoring"
        H  - the Risk dashboard's EWS funnel, precision & lead-time tiles

Snapshot = the latest fully-observable month (config.LAST_FULL_WINDOW_MONTH).
In production the same scoring refreshes on the live month. Band cuts are reused
from Step 8 (policy.thresholds.band_cuts) so the watchlist matches the table.

Run:  python -m policy.segment_overlay
"""

import numpy as np
import pandas as pd

import config as cfg
from policy import thresholds

PRIMARY = "score_scorecard"
DISTRESS_COLS = ["bounce_streak", "bounce_count_3m", "spending_shock_rate_3m", "utilisation_level"]


def _snapshot():
    sc = pd.read_parquet(cfg.OUT / "scores" / "scores.parquet")
    fm = pd.read_parquet(cfg.OUT / "frame" / "feature_matrix_split.parquet")
    cols = [cfg.KEYS["loan"], "months_on_book", "dpd_current"] + DISTRESS_COLS
    sc = sc.merge(fm[cols], on=[cfg.KEYS["loan"], "months_on_book"], how="left")
    snap = sc[sc.obs_month == cfg.LAST_FULL_WINDOW_MONTH].copy()
    return snap


def _assign_band(df, red_cut, amber_cut):
    band = np.where(df.dpd_current >= 30, "ESCALATION",
            np.where(df[PRIMARY] >= red_cut, "RED",
             np.where(df[PRIMARY] >= amber_cut, "AMBER", "GREEN")))
    return band


def _distress(df):
    """E's C4 signature, applied live: persistent bounces, or shock + high utilisation."""
    return ((df.bounce_streak >= 2)
            | (df.bounce_count_3m >= 3)
            | ((df.utilisation_level >= 70) & (df.spending_shock_rate_3m > 0))).astype(int)


def run():
    red_cut, amber_cut = thresholds.band_cuts()
    snap = _snapshot()
    snap["band"] = _assign_band(snap, red_cut, amber_cut)
    snap["is_behavioural_distress"] = _distress(snap)

    print("=" * 74)
    print(f"STEP 9 — SEGMENT OVERLAY + WATCHLIST   (snapshot {cfg.LAST_FULL_WINDOW_MONTH}, "
          f"RED>={red_cut:.3f} AMBER>={amber_cut:.3f})")
    print("=" * 74)
    print(f"loans in snapshot: {len(snap):,}")

    # ---------- 1. flag rates by segment ----------
    print("\nFLAG RATES BY SEGMENT")
    print(f"  {'segment':<32}{'n':>7}{'ESC':>7}{'RED':>7}{'AMBER':>8}{'watch%':>8}")
    seg_rows = []
    for s in cfg.SEGMENTS:
        d = snap[snap.segment == s]
        if not len(d):
            continue
        esc = (d.band == "ESCALATION").mean()
        red = (d.band == "RED").mean()
        amb = (d.band == "AMBER").mean()
        watch = red + amb
        star = "  <- priority" if s in cfg.PRIORITY_SEGMENTS else ""
        print(f"  {s:<32}{len(d):>7,}{esc:>7.1%}{red:>7.1%}{amb:>8.1%}{watch:>8.1%}{star}")
        seg_rows.append({"segment": s, "n": len(d), "pct_escalation": round(esc, 4),
                         "pct_red": round(red, 4), "pct_amber": round(amb, 4),
                         "pct_watchlist": round(watch, 4)})

    # ---------- 2. behavioural-distress cohort ----------
    dc = snap[snap.is_behavioural_distress == 1]
    print(f"\nBEHAVIOURAL-DISTRESS COHORT (live C4 pattern): {len(dc):,} loans "
          f"({len(dc)/len(snap):.1%} of snapshot)")
    if len(dc):
        print(f"  within-window default rate : {dc.label.mean():.1%}  "
              f"vs {snap.label.mean():.1%} snapshot-wide  "
              f"(lift {dc.label.mean()/max(snap.label.mean(),1e-9):.1f}x — echoes E's ~66% C4 group)")
        print(f"  share landing in RED/ESCALATION : {(dc.band.isin(['RED','ESCALATION'])).mean():.0%}")

    # ---------- 3. emit the watchlist ----------
    wl = snap[snap.band.isin(["ESCALATION", "RED", "AMBER"])].copy()
    action = {"ESCALATION": "immediate collections", "RED": "proactive outreach / restructure",
              "AMBER": "watchlist / monitor monthly"}
    wl["recommended_action"] = wl.band.map(action)
    band_rank = {"ESCALATION": 0, "RED": 1, "AMBER": 2}
    wl = wl.sort_values(
        ["is_behavioural_distress", "band", PRIMARY],
        key=lambda c: c.map(band_rank) if c.name == "band" else c,
        ascending=[False, True, False],
    )
    out_cols = [cfg.KEYS["loan"], cfg.KEYS["customer"], "segment", cfg.LOAN_COLS["product"],
                "obs_month", "dpd_current", PRIMARY, "band",
                "is_behavioural_distress", "recommended_action"]
    wl = wl[out_cols].rename(columns={PRIMARY: "ews_score"})
    wl["ews_score"] = wl["ews_score"].round(4)

    print(f"\nWATCHLIST: {len(wl):,} loans  "
          f"(ESC {int((wl.band=='ESCALATION').sum())}, RED {int((wl.band=='RED').sum())}, "
          f"AMBER {int((wl.band=='AMBER').sum())}); sorted distress-first, then band, then score")
    print("  top 8 of the watchlist:")
    print(wl.head(8).to_string(index=False))

    # ---------- done-when ----------
    print("\nDONE-WHEN")
    print(f"  [{'PASS' if seg_rows else 'FAIL'}] segment flag rates tabulated ({len(seg_rows)} segments)")
    print(f"  [{'PASS' if set(['ews_score','band','segment']).issubset(wl.columns) else 'FAIL'}] "
          f"watchlist carries score, band, segment")
    print(f"  [{'PASS' if len(dc) and wl.iloc[0].is_behavioural_distress == 1 else 'CHECK'}] "
          f"behavioural-distress cohort identified and at top")

    # ---------- downstream feed ----------
    print("\nDOWNSTREAM FEED")
    print("  G (optimization): ews_score + band -> tighten limits/pricing on RED before loss")
    print("  I (recommendation): operationalises 'route Contain (#3) to early-warning monitoring'")
    print("  H (dashboard): EWS funnel (book->watch->RED), precision & lead-time tiles")

    out = cfg.OUT_DIRS["thresholds"].parent / "watchlist"; out.mkdir(parents=True, exist_ok=True)
    wl.to_csv(out / "ews_watchlist_current.csv", index=False)
    pd.DataFrame(seg_rows).to_csv(out / "flag_rates_by_segment.csv", index=False)
    print(f"\n  written: {out/'ews_watchlist_current.csv'}, flag_rates_by_segment.csv")
    return wl, seg_rows


if __name__ == "__main__":
    run()
