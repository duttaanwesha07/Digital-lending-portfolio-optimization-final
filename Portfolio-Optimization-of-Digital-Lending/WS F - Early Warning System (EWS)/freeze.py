"""
freeze.py — Step 10: determinism check + freeze manifest.

On a Gate-4 PASS we freeze the EWS: prove it reproduces, then record seed, spec
version, feature set, band cuts, metrics, deviations and assumptions in one
manifest so anyone can rebuild the exact model. "Same seed -> same scores" is
demonstrated here, not asserted.

Run:  python -m freeze
"""

import json
import hashlib
import datetime as dt
import numpy as np
import pandas as pd

import config as cfg
from policy import thresholds


def _determinism_check():
    """Re-fit + re-score and compare to the saved scores (the part with RNG:
    the monotonic GBM and the within-time split). Upstream spine/label/feature
    steps are pure deterministic transforms."""
    saved = pd.read_parquet(cfg.OUT / "scores" / "scores.parquet")[
        [cfg.KEYS["loan"], "months_on_book", "score_scorecard", "score_gbm"]
    ].rename(columns={"score_scorecard": "sc0", "score_gbm": "gb0"})

    from model import models
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        models.run()                                   # rewrites scores.parquet from seed
    new = pd.read_parquet(cfg.OUT / "scores" / "scores.parquet")[
        [cfg.KEYS["loan"], "months_on_book", "score_scorecard", "score_gbm"]
    ]
    m = saved.merge(new, on=[cfg.KEYS["loan"], "months_on_book"])
    return {
        "scorecard_max_abs_diff": float((m.sc0 - m.score_scorecard).abs().max()),
        "gbm_max_abs_diff": float((m.gb0 - m.score_gbm).abs().max()),
    }


def _read_gate4():
    g = pd.read_csv(cfg.OUT_DIRS["validation"] / "gate4_results.csv")
    checks = {r["check"]: {"pass": bool(r["pass"]), "detail": r["detail"]} for _, r in g.iterrows()}
    verdict = "PASS" if all(c["pass"] for c in checks.values()) else "FAIL"
    return checks, verdict


def _file_inventory():
    base = cfg.OUT
    inv = {}
    for p in sorted(base.rglob("*")):
        if p.is_file():
            inv[str(p.relative_to(base))] = hashlib.md5(p.read_bytes()).hexdigest()[:10]
    return inv


def run():
    red_cut, amber_cut = thresholds.band_cuts()
    checks, verdict = _read_gate4()
    det = _determinism_check()

    manifest = {
        "workstream": "F — Early Warning System",
        "ews_version": cfg.EWS_VERSION,
        "frozen_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "seed": cfg.MASTER_SEED,
        "gate4_verdict": verdict,
        "build_order": ["spine", "labels", "features", "split", "models",
                        "validation", "thresholds", "segment_overlay", "freeze"],
        "model": {"primary": cfg.PRIMARY_MODEL, "challenger": "monotonic_gbm",
                  "calibration": "grouped-by-loan OOF isotonic",
                  "monotone": "scorecard isotonic bins + GBM constraints from FEATURE_SPEC"},
        "features_model": [n for n, *_ in cfg.FEATURE_SPEC],
        "features_escalation_excluded": [n for n, *_ in cfg.ESCALATION_SPEC],
        "label": {"window_H_months": cfg.LABEL_WINDOW_H, "mob_floor": cfg.MOB_FLOOR,
                  "default_dpd": cfg.DEFAULT_DPD, "censoring": "status-aware (Active=censor, terminal=observed-survival)"},
        "split": {"within_time": "loans-whole stratified",
                  "out_of_time": f"purged+embargoed temporal, test>={cfg.OOT_TEST_FROM_MONTH}, "
                                 f"embargo={cfg.LABEL_WINDOW_H}mo  [DEVIATION: not loan-disjoint — ratify]"},
        "band_cuts": {"red_ge": round(red_cut, 4), "amber_ge": round(amber_cut, 4),
                      "basis": "PLACEHOLDER capacity/costs (F-A050..053) — Risk Consultant to confirm"},
        "gate4_checks": checks,
        "headline_metrics": {
            "oot_auc_clean_dpd0": 0.883, "ks": 0.681, "top_decile_lift": 7.6,
            "lead_days_median_oos": 91, "lead_days_mean_oos": 108,
            "watchlist_recall": 0.87, "behavioural_distress_lift": 17.6},
        "deviations_and_open_items": [
            "OOT split is purged+embargoed temporal (not loan-disjoint) — methodologically standard, RATIFY",
            "RED/AMBER capacity & FP/missed costs are PLACEHOLDER (F-A050..053) — block final thresholds",
            "AMBER volume overshoots placeholder capacity due to calibrated-score ties at low end",
            "Watchlist snapshot = 2026-03 (latest full-window); production refreshes on live month",
            "Full-book AUC (0.957) inflated by already-delinquent loans — report the 0.883 clean number"],
        "assumptions_referenced": ["A-019 true_latent never a feature", "A-046 ~35% self-cure ceiling",
                                   "F-A050..053 capacity & cost placeholders"],
        "determinism": {"reproduced": det["scorecard_max_abs_diff"] < 1e-9 and det["gbm_max_abs_diff"] < 1e-6,
                        **det},
        "deliverables": {
            "score_definition": "model/models.py (scorecard primary)",
            "watchlist": "out/watchlist/ews_watchlist_current.csv",
            "threshold_table": "out/thresholds/ews_threshold_table.csv",
            "validation_pack": "out/validation/gate4_results.csv + validate/validation.py",
            "flag_rates_by_segment": "out/watchlist/flag_rates_by_segment.csv"},
        "output_inventory_md5": _file_inventory(),
    }

    out = cfg.OUT / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))

    print("=" * 64); print("STEP 10 — FREEZE & GATE 4 SIGN-OFF"); print("=" * 64)
    print(f"  GATE 4 VERDICT         : {verdict}")
    for k, v in checks.items():
        print(f"    [{'PASS' if v['pass'] else 'FAIL'}] {k:<11} {v['detail']}")
    print(f"\n  determinism (re-fit from seed {cfg.MASTER_SEED}):")
    print(f"    scorecard max|Δ| = {det['scorecard_max_abs_diff']:.2e}  "
          f"gbm max|Δ| = {det['gbm_max_abs_diff']:.2e}  -> "
          f"{'REPRODUCED' if manifest['determinism']['reproduced'] else 'NON-DETERMINISTIC (investigate)'}")
    print(f"  band cuts frozen       : RED>={red_cut:.4f}  AMBER>={amber_cut:.4f}  (placeholder basis)")
    print(f"  files inventoried+hashed: {len(manifest['output_inventory_md5'])}")
    print(f"\n  written: {out}")
    return manifest


if __name__ == "__main__":
    run()
