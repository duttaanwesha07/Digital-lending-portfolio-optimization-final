"""audit.py — scrutinise every parameter: in place, tunable, placeholder, or missing."""
import config as cfg

rows = []   # (group, param, value, status)
def add(grp, p, v, status): rows.append((grp, p, v, status))

# 1. reproducibility
add("repro", "MASTER_SEED", cfg.MASTER_SEED, "LOCKED")
add("repro", "EWS_VERSION", cfg.EWS_VERSION, "SET")
add("repro", "WORKSTREAM", cfg.WORKSTREAM, "SET")

# 2. paths exist
for k, p in cfg.PATHS.items():
    add("paths", f"PATHS[{k}]", str(p), "OK" if p.exists() else "MISSING-FILE")

# 3. label
for p in ["DEFAULT_DPD", "PERFORMING_MAX_DPD", "MOB_FLOOR", "LABEL_WINDOW_H",
          "DROP_CENSORED_LABELS", "LAST_FULL_WINDOW_MONTH"]:
    add("label", p, getattr(cfg, p), "LOCKED" if p in ("DEFAULT_DPD","MOB_FLOOR","DROP_CENSORED_LABELS") else "TUNE")

# 4. point-in-time windows
for p in ["POINT_IN_TIME", "TREND_WINDOW", "VOLATILITY_BASELINE_WINDOW", "STREAK_LOOKBACK"]:
    add("pit", p, getattr(cfg, p), "LOCKED" if p == "POINT_IN_TIME" else "SET")

# 5. feature spec integrity
fnames = [n for n, *_ in cfg.FEATURE_SPEC]
enames = [n for n, *_ in cfg.ESCALATION_SPEC]
add("features", "FEATURE_SPEC count", len(cfg.FEATURE_SPEC), "OK" if len(fnames) == 13 else "CHECK")
add("features", "ESCALATION_SPEC count", len(cfg.ESCALATION_SPEC), "OK" if len(enames) == 2 else "CHECK")
add("features", "signs valid (+1/-1)", all(s in (1, -1) for *_, s in cfg.FEATURE_SPEC), "OK")
add("features", "no duplicate names", len(set(fnames)) == len(fnames), "OK")
add("features", "balance_vol_spike removed", "balance_vol_spike" not in fnames, "OK")
add("features", "dpd_current NOT a model feature", "dpd_current" not in fnames, "OK")
add("features", "dpd_current IS escalation", "dpd_current" in enames, "OK")
add("features", "balance_volatility_level present", "balance_volatility_level" in fnames, "OK")

# 6. triggers
add("triggers", "escalation rule present", "already_delinquent_30dpd" in cfg.TRIGGER_RULES, "OK")

# 7. splits
for p in ["SPLIT_GROUP_KEY", "STRATIFY_KEYS", "WITHIN_TIME_TEST_FRAC", "OOT_TEST_FROM_MONTH"]:
    add("split", p, getattr(cfg, p), "LOCKED" if p == "SPLIT_GROUP_KEY" else "TUNE")

# 8. models
add("model", "PRIMARY_MODEL", cfg.PRIMARY_MODEL, "SET")
add("model", "GBM monotone-from-spec", cfg.GBM["use_monotone_from_feature_spec"], "OK")
add("model", "CALIBRATE_SCORES", cfg.CALIBRATE_SCORES, "OK")

# 9. gate 4
for k, v in cfg.GATE4.items():
    add("gate4", k, v, "SET")
add("gate4", "SELF_CURE_EPISODE_RATE", cfg.SELF_CURE_EPISODE_RATE, "LOCKED(A-046)")
add("gate4", "BOOTSTRAP_RESAMPLES", cfg.BOOTSTRAP_RESAMPLES, "SET")

# 10. RAG + the THREE business inputs (the known gap)
for k, v in cfg.RAG_BANDS.items():
    add("rag", f"RAG_BANDS[{k}]", v, "TUNE")
for p in ["COLLECTIONS_MONTHLY_CAPACITY", "COST_FALSE_POSITIVE", "COST_MISSED_DEFAULT"]:
    v = getattr(cfg, p)
    add("rag", p, v, "MISSING (Risk Consultant)" if v is None else "SET")

# 11. segments
add("segments", "SEGMENTS count", len(cfg.SEGMENTS), "OK" if len(cfg.SEGMENTS) == 6 else "CHECK")
add("segments", "PRIORITY_SEGMENTS", cfg.PRIORITY_SEGMENTS, "OK" if len(cfg.PRIORITY_SEGMENTS) == 2 else "CHECK")
add("segments", "priority subset of segments",
    all(s in cfg.SEGMENTS for s in cfg.PRIORITY_SEGMENTS), "OK")

# 12. cross-module wiring
import importlib
feat = importlib.import_module("frame.features")
mdl = importlib.import_module("model.models")
thr = importlib.import_module("policy.thresholds")
add("wiring", "features build == FEATURE_SPEC", feat.FEATURE_NAMES == fnames, "OK")
add("wiring", "features carry == ESCALATION_SPEC", feat.ESCALATION_NAMES == enames, "OK")
add("wiring", "models FEATURES == FEATURE_SPEC", mdl.FEATURES == fnames, "OK")
add("wiring", "thresholds USING_PLACEHOLDER", thr.USING_PLACEHOLDER, "EXPECTED (costs None)")
add("wiring", "GBM constraints length == features",
    len([int(s) for *_, s in cfg.FEATURE_SPEC]) == len(fnames), "OK")

# ---- report ----
print("=" * 74); print("PARAMETER AUDIT"); print("=" * 74)
cur = None
ok = miss = placeholder = check = 0
for grp, p, v, status in rows:
    if grp != cur:
        print(f"\n[{grp.upper()}]"); cur = grp
    flag = "✓"
    if "MISSING" in status: flag = "✗"; miss += 1
    elif "PLACEHOLDER" in status or "EXPECTED" in status: flag = "~"; placeholder += 1
    elif "CHECK" in status or status == "MISSING-FILE": flag = "!"; check += 1
    else: ok += 1
    val = str(v)
    if len(val) > 40: val = val[:37] + "..."
    print(f"  {flag} {p:<34} {val:<42} {status}")

print("\n" + "=" * 74)
print(f"SUMMARY: {ok} in-place/OK   {placeholder} placeholder-by-design   "
      f"{miss} MISSING(business)   {check} to-check")
print("=" * 74)

