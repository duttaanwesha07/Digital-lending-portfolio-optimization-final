"""
policy/thresholds.py — Step 8: green / amber / red bands.

Turns the calibrated EWS score into actions collections can run. Two real
constraints set the cuts (never a round number):

  COST ASYMMETRY  -> the break-even precision a RED flag must clear:
        p* = cost_FP / (cost_FP + cost_missed_default)
     A missed 90+ costs far more than a soft-touch call, so p* is low — meaning
     intervention is worthwhile even at modest precision. This sets the FLOOR.

  COLLECTIONS CAPACITY -> how many RED (intensive) and AMBER (watch) flags the
     team can work per month. This sets the VOLUME, and is the binding constraint.

Escalation tier: loans already >=30 DPD bypass the bands — they go straight to
collections via the `already_delinquent_30dpd` trigger, not the early-warning score.

*** BLOCKED INPUT — the three business numbers below are the Risk Consultant's.
    Until they land in config (currently None), this module uses clearly-flagged
    PLACEHOLDERS and logs them as assumptions. The moment config carries real
    values, they override automatically and every number recomputes. ***

Run:  python -m policy.thresholds
"""

import numpy as np
import pandas as pd

import config as cfg

PRIMARY = "score_scorecard"
DAYS = 30.44

# --- business inputs: config first, PLACEHOLDER only while config is None -----
PLACEHOLDER = {
    "collections_red_per_month":   500,    # F-A050 PLACEHOLDER — intensive-work capacity
    "collections_amber_per_month": 1500,   # F-A051 PLACEHOLDER — watch-tier capacity
    "cost_false_positive":         300,    # F-A052 PLACEHOLDER — INR, soft-touch outreach
    "cost_missed_default":       25000,    # F-A053 PLACEHOLDER — INR, avg loss on a missed 90+
}
RED_CAP   = cfg.COLLECTIONS_MONTHLY_CAPACITY or PLACEHOLDER["collections_red_per_month"]
AMBER_CAP = PLACEHOLDER["collections_amber_per_month"]      # config has no separate amber cap yet
COST_FP   = cfg.COST_FALSE_POSITIVE or PLACEHOLDER["cost_false_positive"]
COST_MISS = cfg.COST_MISSED_DEFAULT or PLACEHOLDER["cost_missed_default"]
USING_PLACEHOLDER = (cfg.COLLECTIONS_MONTHLY_CAPACITY is None
                     or cfg.COST_FALSE_POSITIVE is None
                     or cfg.COST_MISSED_DEFAULT is None)


def _load():
    sc = pd.read_parquet(cfg.OUT / "scores" / "scores.parquet")
    fm = pd.read_parquet(cfg.OUT / "frame" / "feature_matrix_split.parquet")
    sc = sc.merge(fm[[cfg.KEYS["loan"], "months_on_book", "dpd_current"]],
                  on=[cfg.KEYS["loan"], "months_on_book"], how="left")
    sc["obs_p"] = pd.PeriodIndex(sc["obs_month"], freq="M")
    return sc


def _lead_days(op_def_rows, D_ord_map, thr):
    """OOS mean lead (days) for pre-escalation rows scoring >= thr."""
    w = op_def_rows[op_def_rows[PRIMARY] >= thr]
    if not len(w):
        return 0.0, 0
    first = w.sort_values("obs_ord").groupby(cfg.KEYS["loan"]).first()
    lead = (np.array([D_ord_map[l] for l in first.index]) - first["obs_ord"].values) * DAYS
    return float(np.mean(lead)), len(first)


def band_cuts():
    """RAG score cut-offs, derived from capacity on the test operating book.
    Reused by Step 9 so the watchlist bands match the threshold table exactly."""
    sc = _load()
    te = sc[sc.split_oot == "test"]
    n_months = te["obs_month"].nunique()
    op = te[te.dpd_current < 30]
    op_monthly = len(op) / n_months
    red_frac = min(RED_CAP / op_monthly, 1.0)
    amber_frac = min(AMBER_CAP / op_monthly, 1.0 - red_frac)
    return (op[PRIMARY].quantile(1 - red_frac),
            op[PRIMARY].quantile(1 - red_frac - amber_frac))


def run():
    sc = _load()
    te = sc[sc.split_oot == "test"].copy()
    n_months = te["obs_month"].nunique()

    op = te[te.dpd_current < 30].copy()                 # operating book (pre-escalation)
    esc = te[te.dpd_current >= 30]                       # escalation tier (bypass bands)
    op_monthly = len(op) / n_months

    # --- cost asymmetry -> break-even precision (the RED floor) ---
    p_star = COST_FP / (COST_FP + COST_MISS)

    # --- capacity -> band fractions on the operating book ---
    red_cut, amber_cut = band_cuts()

    # --- per-band metrics on the operating book ---
    def band_stats(mask):
        d = op[mask]
        n = len(d); pos = int(d.label.sum())
        prec = pos / n if n else 0.0
        rec = pos / max(int(op.label.sum()), 1)
        return n, n / n_months, prec, rec

    red_m = op[PRIMARY] >= red_cut
    amb_m = (op[PRIMARY] >= amber_cut) & (op[PRIMARY] < red_cut)
    grn_m = op[PRIMARY] < amber_cut

    # lead time (OOS) for RED and for the RED+AMBER watchlist
    rep = pd.read_parquet(cfg.PATHS["repayments"])
    Dcal = rep[rep.dpd >= cfg.DEFAULT_DPD].groupby(cfg.KEYS["loan"])["period_date"].min()
    D_ord = pd.Series(pd.PeriodIndex(Dcal, freq="M").map(lambda p: p.ordinal), index=Dcal.index)
    cut = pd.Period(cfg.OOT_TEST_FROM_MONTH, freq="M")
    D_ord_map = D_ord.to_dict()
    test_def = set(D_ord[D_ord >= cut.ordinal].index)
    op["obs_ord"] = op["obs_p"].apply(lambda p: p.ordinal)
    op_def = op[op[cfg.KEYS["loan"]].isin(test_def)]
    red_lead, _ = _lead_days(op_def, D_ord_map, red_cut)
    watch_lead, _ = _lead_days(op_def, D_ord_map, amber_cut)

    # ---------- report ----------
    print("=" * 72); print("STEP 8 — RAG BAND / THRESHOLD DESIGN"); print("=" * 72)
    if USING_PLACEHOLDER:
        print("  *** PLACEHOLDER business inputs in use (Risk Consultant to confirm) ***")
    print(f"  cost_FP=₹{COST_FP:,}  cost_missed=₹{COST_MISS:,}  ->  break-even precision p* = {p_star:.2%}")
    print(f"  capacity: RED {RED_CAP:,}/mo, AMBER {AMBER_CAP:,}/mo  | operating book ≈ {op_monthly:,.0f}/mo")
    print(f"  derived cuts on calibrated score:  RED >= {red_cut:.4f}   AMBER >= {amber_cut:.4f}")

    print("\n  band            score range        vol/mo    % book   precision   recall   mean lead   action")
    rows = []
    def line(name, rng, stats, lead, action):
        n, vm, prec, rec = stats
        lead_s = f"{lead:>6.0f}d" if lead else "    — "
        print(f"  {name:<14}{rng:<18}{vm:>8.0f}{n/len(op):>9.1%}{prec:>11.1%}{rec:>9.1%}{lead_s:>11}   {action}")
        rows.append({"band": name, "score_range": rng, "vol_per_month": round(vm),
                     "pct_book": round(n/len(op), 4), "precision": round(prec, 4),
                     "recall": round(rec, 4), "mean_lead_days": round(lead), "action": action})

    # escalation tier first (not part of the bands)
    esc_pos = int(esc.label.sum())
    print(f"  {'ESCALATION':<14}{'dpd >= 30':<18}{len(esc)/n_months:>8.0f}{'—':>9}"
          f"{(esc_pos/len(esc) if len(esc) else 0):>11.1%}{'—':>9}{'now':>11}   immediate collections")
    rows.append({"band": "ESCALATION", "score_range": "dpd>=30", "vol_per_month": round(len(esc)/n_months),
                 "pct_book": None, "precision": round(esc_pos/len(esc), 4) if len(esc) else 0,
                 "recall": None, "mean_lead_days": 0, "action": "immediate collections"})
    line("RED",   f">= {red_cut:.3f}", band_stats(red_m), red_lead, "proactive outreach / restructure")
    line("AMBER", f"{amber_cut:.3f}–{red_cut:.3f}", band_stats(amb_m), 0, "watchlist / monitor monthly")
    line("GREEN", f"< {amber_cut:.3f}", band_stats(grn_m), 0, "no action")

    # cost-justification check
    red_prec = band_stats(red_m)[2]
    print(f"\n  cost check: RED precision {red_prec:.2%}  vs break-even p* {p_star:.2%}  -> "
          f"{'JUSTIFIED' if red_prec >= p_star else 'NOT justified (raise cut)'}")
    watch_n, watch_vm, _, watch_rec = (
        len(op[red_m | amb_m]), len(op[red_m | amb_m]) / n_months, 0,
        int(op[red_m | amb_m].label.sum()) / max(int(op.label.sum()), 1))
    print(f"  watchlist (RED+AMBER): {watch_vm:,.0f}/mo, catches {watch_rec:.0%} of operating-book "
          f"defaults, mean lead {watch_lead:.0f}d")
    print(f"\n  NOTE: ~{cfg.SELF_CURE_EPISODE_RATE:.0%} of stress episodes self-resolve (A-046) — a share of "
          f"AMBER will cure without defaulting; that is the intended watch-tier behaviour, not error.")

    # ---------- write outputs (one number, one source) ----------
    out = cfg.OUT_DIRS["thresholds"]; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "ews_threshold_table.csv", index=False)

    headlines = pd.DataFrame([{
        "metric": "flag_volume_monthly_red", "value": round(band_stats(red_m)[1]),
        "source": "WS-F Step 8", "status": "PLACEHOLDER" if USING_PLACEHOLDER else "final"},
        {"metric": "flag_volume_monthly_watchlist", "value": round(watch_vm),
         "source": "WS-F Step 8", "status": "PLACEHOLDER" if USING_PLACEHOLDER else "final"},
        {"metric": "precision_at_red", "value": round(red_prec, 4),
         "source": "WS-F Step 8", "status": "PLACEHOLDER" if USING_PLACEHOLDER else "final"},
        {"metric": "lead_time_days_mean_red", "value": round(red_lead),
         "source": "WS-F Step 8", "status": "PLACEHOLDER" if USING_PLACEHOLDER else "final"},
    ])
    headlines.to_csv(out / "controlled_headlines.csv", index=False)

    # assumptions log (placeholders, project convention)
    pd.DataFrame([
        {"id": "F-A050", "assumption": "collections RED capacity/month", "value": RED_CAP,
         "status": "PLACEHOLDER" if USING_PLACEHOLDER else "confirmed", "rationale": "intensive-work team limit"},
        {"id": "F-A051", "assumption": "collections AMBER capacity/month", "value": AMBER_CAP,
         "status": "PLACEHOLDER", "rationale": "watch-tier team limit"},
        {"id": "F-A052", "assumption": "cost of false positive (INR)", "value": COST_FP,
         "status": "PLACEHOLDER" if USING_PLACEHOLDER else "confirmed", "rationale": "soft-touch outreach cost"},
        {"id": "F-A053", "assumption": "cost of missed default (INR)", "value": COST_MISS,
         "status": "PLACEHOLDER" if USING_PLACEHOLDER else "confirmed", "rationale": "avg loss on a missed 90+"},
    ]).to_csv(out / "assumptions_F_step8.csv", index=False)

    print(f"\n  written: {out/'ews_threshold_table.csv'}, controlled_headlines.csv, assumptions_F_step8.csv")
    return rows, headlines


if __name__ == "__main__":
    run()
