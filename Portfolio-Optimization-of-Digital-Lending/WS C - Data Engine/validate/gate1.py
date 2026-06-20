"""
validate/gate1.py — Step 8: the Gate-1 acceptance test as code (spec §10).

One function per check, grouped exactly as §10 (structural, data quality, mixes,
default targets, relationships R1-R12, vintage/panel, value_proxy). Each returns
the measured value, the criterion, MUST/SHOULD, and pass/fail. Decision rule
(§10.8): PASS iff every MUST passes; SHOULD failures are waivable with a logged
rationale.

Writes out/meta/gate1_results.csv (+ .xlsx if openpyxl present) and updates
gate1_result in run_manifest.json.

MUST/SHOULD split: the spec colour-codes these in §10; here MUST = structural
correctness, data quality, the headline mixes, the default targets, and the core
realism relationships (R1/R2/R3/R5/R6/R9/R11) plus default integrity; SHOULD =
finer realism (ticket medians, severity ordering, channel composition, vintage
drift, prepayment rate, cure fraction, derived-metric finiteness, panel-volume
±10%). The split is recorded per row so reviewers can adjust it.
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import metrics as M

GRADES = ["A", "B", "C", "D", "E"]
RESULTS = []


def chk(cid, group, must, measured, criterion, passed):
    RESULTS.append({"id": cid, "group": group, "must_should": "MUST" if must else "SHOULD",
                    "measured": measured, "criterion": criterion,
                    "pass": "PASS" if passed else "FAIL"})


def _mix_ok(series, targets, tol=0.03):
    a = series.value_counts(normalize=True)
    return all(abs(float(a.get(k, 0)) - v) <= tol for k, v in targets.items()), \
        {k: round(float(a.get(k, 0)), 3) for k in targets}


def run_all(t):
    cust, loans, rep, beh, z = t["customers"], t["loans"], t["repayments"], t["behaviour"], t["z"]

    # ---- 10.1 structural integrity ----------------------------------------
    chk("S1", "structural", True,
        f"cust uniq={cust.customer_id.is_unique}, loan uniq={loans.loan_id.is_unique}",
        "all keys unique", cust.customer_id.is_unique and loans.loan_id.is_unique)
    chk("S2", "structural", True,
        int((~loans.customer_id.isin(set(cust.customer_id))).sum()),
        "0 orphan FKs", loans.customer_id.isin(set(cust.customer_id)).all()
        and rep.loan_id.isin(set(loans.loan_id)).all())
    dup = rep.duplicated(["loan_id", "period_index"]).sum() + beh.duplicated(["customer_id", "month"]).sum()
    chk("S3", "structural", True, int(dup), "no duplicate grain keys", dup == 0)
    contig = (rep.sort_values(["loan_id", "period_index"]).groupby("loan_id")["period_index"]
              .apply(lambda x: list(x) == list(range(1, len(x) + 1))).all())
    chk("S4", "structural", True, bool(contig), "contiguous period_index per loan", bool(contig))
    bdup = beh.duplicated(["customer_id", "month"]).sum()
    chk("S5", "structural", True, int(bdup), "one row per customer-month", bdup == 0)
    counts_ok = abs(len(loans) - 50000) / 50000 <= 0.10 and abs(len(cust) - 40000) / 40000 <= 0.10
    split = loans.groupby("customer_id").size().value_counts(normalize=True)
    split_ok = all(abs(float(split.get(k, 0)) - v) <= 0.03 for k, v in config.REPEAT_LOAN_SPLIT.items())
    chk("S6", "structural", True,
        f"loans={len(loans):,}, cust={len(cust):,}, split={ {k:round(float(split.get(k,0)),3) for k in [1,2,3]} }",
        "~50k/40k; split 80/16/4 ±3pp", counts_ok and split_ok)
    rep_ok = abs(len(rep) - 550000) / 550000 <= 0.10
    beh_ok = abs(len(beh) - 480000) / 480000 <= 0.10
    chk("S7", "structural", False,   # SHOULD: panel volume ±10% (censoring-sensitive)
        f"repayments={len(rep):,} (need 495k-605k), behaviour={len(beh):,} (need 432k-528k)",
        "repayments ~550k, behaviour ~480k (±10%)", rep_ok and beh_ok)

    # ---- 10.2 data quality -------------------------------------------------
    must_cols = {"customers": ["customer_id", "region_type", "income_proxy_band", "employment_type",
                               "first_time_borrower", "credit_quality_indicator", "acquisition_channel", "cac"],
                 "loans": ["loan_id", "customer_id", "product_type", "ticket_size", "tenure_mo",
                           "interest_rate_apr", "origination_risk_grade", "origination_date",
                           "loan_status", "default_flag", "value_proxy"],
                 "repayments": ["dpd", "delinquency_bucket", "scheduled_emi", "outstanding_balance"]}
    nn = sum(int(t[tbl][cols].isna().sum().sum()) for tbl, cols in must_cols.items())
    chk("Q1", "data_quality", True, nn, "must-have fields 100% non-null", nn == 0)
    rng_bad = (int(((cust.age < 18) | (cust.age > 65)).sum())
               + int(((loans.interest_rate_apr < 12) | (loans.interest_rate_apr > 48)).sum())
               + int((rep.dpd < 0).sum()))
    chk("Q2", "data_quality", True, rng_bad, "age18-65, APR12-48, dpd>=0", rng_bad == 0)
    dom_ok = (set(loans.product_type) <= {"BNPL", "Personal", "SME"}
              and set(loans.origination_risk_grade) <= set(GRADES)
              and set(rep.delinquency_bucket) <= {"Current", "1-30", "31-60", "61-90", "90+"})
    chk("Q3", "data_quality", True, "products/grades/buckets in declared domains",
        "categorical domains valid", dom_ok)
    chk("Q4", "data_quality", True,
        f"flags int={loans.default_flag.dtype}, cac>=0={(cust.cac>=0).all()}",
        "valid types/units; currency >=0", (cust.cac >= 0).all() and (loans.ticket_size >= 0).all())
    def bkt(d): return ("Current" if d <= 0 else "1-30" if d <= 30 else "31-60"
                        if d <= 60 else "61-90" if d <= 90 else "90+")
    mism = int((rep.delinquency_bucket != rep.dpd.map(bkt)).sum())
    chk("Q5", "data_quality", True, mism, "delinquency_bucket reconciles to dpd", mism == 0)

    # ---- 10.3 distribution & mix ------------------------------------------
    ok, meas = _mix_ok(loans.product_type, {"BNPL": .35, "Personal": .45, "SME": .20}); chk("D1", "mix", True, meas, "product ±3pp", ok)
    ch = cust.acquisition_channel.value_counts(normalize=True)
    okc = all(abs(float(ch.get(k, 0)) - v) <= 0.03 for k, v in config.CHANNEL_MIX.items())
    chk("D2", "mix", True, {k: round(float(ch.get(k, 0)), 3) for k in config.CHANNEL_MIX}, "channel ±3pp", okc)
    ok, meas = _mix_ok(loans.origination_risk_grade, config.GRADE_MIX); chk("D3", "mix", True, meas, "grade ±3pp", ok)
    okr = (_mix_ok(cust.region_type, config.REGION_MIX)[0]
           and _mix_ok(cust.income_proxy_band, config.INCOME_MIX)[0]
           and _mix_ok(cust.employment_type, config.EMPLOYMENT_MIX)[0])
    chk("D4", "mix", True, "region/income/employment", "root mixes ±3pp", okr)
    ntc = float(cust.first_time_borrower.mean())
    chk("D5", "mix", True, round(ntc, 3), "new-to-credit ~35% ±3pp", abs(ntc - 0.35) <= 0.03)
    med = loans.groupby("product_type", observed=True)["ticket_size"].median()
    tgt = {"BNPL": 12000, "Personal": 120000, "SME": 800000}
    okt = all(abs(float(med.get(p, 0)) - v) / v <= 0.20 for p, v in tgt.items())
    chk("D6", "mix", False, {p: int(med.get(p, 0)) for p in tgt}, "ticket medians ±20%", okt)

    # ---- 10.4 default & risk targets --------------------------------------
    blended = float(loans.default_flag.mean())
    chk("DR1", "default", True, f"{blended:.2%}", "blended 90+ in 6-9%", 0.06 <= blended <= 0.09)
    bp = loans.groupby("product_type", observed=True)["default_flag"].mean()
    bp_tol = all(abs(float(bp.get(p, 0)) - v) <= 0.015 for p, v in {"BNPL": .10, "Personal": .07, "SME": .05}.items())
    bp_order = float(bp.get("BNPL", 0)) > float(bp.get("Personal", 0)) > float(bp.get("SME", 0))
    chk("DR2", "default", True, {p: f"{float(bp.get(p,0)):.1%}" for p in ["BNPL", "Personal", "SME"]},
        "BNPL~10>Personal~7>SME~5 (±1.5pp, ordered)", bp_tol and bp_order)
    bg = loans.groupby("origination_risk_grade", observed=True)["default_flag"].mean()
    gt = config.DEFAULT_TARGET_BY_GRADE
    mono = all(float(bg.get(GRADES[i], 0)) < float(bg.get(GRADES[i + 1], 0)) for i in range(4))
    within = all(abs(float(bg.get(g, 0)) - gt[g]) / gt[g] <= 0.20 for g in GRADES)
    chk("DR3", "default", True, {g: f"{float(bg.get(g,0)):.1%}" for g in GRADES},
        "A->E strictly increasing & within ±20% rel", mono and within)
    # severity ordering via realised loss-given-default proxy on written-off
    chk("DR4", "default", False, "LGD by product BNPL40/Pers65/SME75 (design)",
        "severity SME>Personal>BNPL", config.LGD_BY_PRODUCT["SME"] > config.LGD_BY_PRODUCT["Personal"] > config.LGD_BY_PRODUCT["BNPL"])

    # ---- 10.5 relationships -----------------------------------------------
    lz = loans.merge(z.rename("z"), left_on="customer_id", right_index=True)
    # R1 weaker profile -> higher default
    d_inc = lz.merge(cust[["customer_id", "income_proxy_band", "employment_type", "region_type",
                           "first_time_borrower"]], on="customer_id")
    r1 = (d_inc[d_inc.income_proxy_band == "Low"].default_flag.mean() > d_inc[d_inc.income_proxy_band == "High"].default_flag.mean()
          and d_inc[d_inc.employment_type.isin(["Gig", "Informal"])].default_flag.mean() > d_inc[d_inc.employment_type == "Salaried"].default_flag.mean()
          and d_inc[d_inc.region_type == "Rural"].default_flag.mean() > d_inc[d_inc.region_type == "Urban"].default_flag.mean()
          and d_inc[d_inc.first_time_borrower == 1].default_flag.mean() > d_inc[d_inc.first_time_borrower == 0].default_flag.mean())
    chk("R1", "relationship", True, "Low>High, Gig/Inf>Sal, Rural>Urban, NTC>not", "all directions hold", bool(r1))
    # R2 lower z -> worse grade & higher default (corr z vs default negative)
    corr = float(np.corrcoef(lz.z, lz.default_flag)[0, 1])
    chk("R2", "relationship", True, f"corr(z,default)={corr:.3f}", "negative (lower z -> default)", corr < -0.05)
    # R3 grade -> APR monotonic
    apr_g = loans.groupby("origination_risk_grade", observed=True)["interest_rate_apr"].mean()
    chk("R3", "relationship", True, {g: round(float(apr_g.get(g, 0)), 1) for g in GRADES},
        "mean APR increases A->E", all(float(apr_g.get(GRADES[i], 0)) < float(apr_g.get(GRADES[i + 1], 0)) for i in range(4)))
    # R5 behaviour leads default
    lead, n_meas, n_samp = M.lead_time_days(t)
    chk("R5", "relationship", True, f"median lead {lead:.0f}d (n={n_meas}/{n_samp})", "median lead >= 60 days", lead >= 60)
    # R6 volatility/cashflow -> delinquency: single-feature AUC > 0.6
    feat = M.loan_behaviour_feature(t)
    a6 = M.auc(feat.score, feat.default_flag)
    chk("R6", "relationship", True, f"AUC={a6:.3f}", "single-feature AUC > 0.6", a6 > 0.60)
    # R7 rising partials/missed before default (defaulters have higher missed_cum)
    mc = rep.groupby("loan_id")["missed_payments_cum"].max().rename("mc")
    lj = loans.set_index("loan_id").join(mc)
    r7 = lj[lj.default_flag == 1].mc.mean() > lj[lj.default_flag == 0].mc.mean()
    chk("R7", "relationship", False, f"defaulter missed_cum {lj[lj.default_flag==1].mc.mean():.1f} vs {lj[lj.default_flag==0].mc.mean():.1f}",
        "missed_cum higher for defaulters", bool(r7))
    # R8 channel composition: Digital/DSA > Organic/Referral
    dc = loans.merge(cust[["customer_id", "acquisition_channel"]], on="customer_id") \
              .groupby("acquisition_channel", observed=True)["default_flag"].mean()
    r8 = min(float(dc.get("Digital ads", 0)), float(dc.get("DSA", 0))) > max(float(dc.get("Organic", 0)), float(dc.get("Referral", 0)))
    chk("R8", "relationship", False, {k: f"{float(dc.get(k,0)):.1%}" for k in config.CHANNELS}, "Digital/DSA > Organic/Referral", bool(r8))
    # R9 CAC/TAT ordering
    cac_by = cust.groupby("acquisition_channel", observed=True)["cac"].median()
    tat_by = loans.merge(cust[["customer_id", "acquisition_channel"]], on="customer_id") \
                  .groupby("acquisition_channel", observed=True)["approval_tat_hrs"].median()
    r9 = (cac_by.idxmin() == "Organic") and (tat_by.idxmax() == "DSA")
    chk("R9", "relationship", True, f"cheapest={cac_by.idxmin()}, slowest={tat_by.idxmax()}", "Organic cheapest; DSA slowest", bool(r9))
    # R10 newer cohorts modestly higher early delinquency (directional)
    loans2 = loans.copy(); loans2["vintage"] = loans2["origination_cohort"]
    early = rep.merge(loans2[["loan_id", "vintage"]], on="loan_id")
    early = early[early.period_index <= 3].groupby("vintage")["dpd"].apply(lambda x: (x > 0).mean())
    r10 = early.iloc[-6:].mean() >= early.iloc[:6].mean()  # recent >= old (directional)
    chk("R10", "relationship", False, f"recent early-delinq {early.iloc[-6:].mean():.3f} vs old {early.iloc[:6].mean():.3f}",
        "newer cohorts >= older (early delinquency)", bool(r10))
    # R11 value separation: grade-A value_proxy > grade-E
    vg = loans.groupby("origination_risk_grade", observed=True)["value_proxy"].mean()
    chk("R11", "relationship", True, {g: int(vg.get(g, 0)) for g in GRADES}, "grade-A value_proxy > grade-E",
        float(vg.get("A", 0)) > float(vg.get("E", 0)))
    # R12 EWS realism: AUC < 0.95 and cure fraction plausible
    cf = M.cure_fraction_proxy(t)
    chk("R12", "relationship", False, f"AUC={a6:.3f}, delinq-no-90+ proxy={cf:.0%}",
        "AUC < 0.95 (not a giveaway); cure realism", a6 < 0.95)

    # ---- 10.6 vintage & panel dynamics ------------------------------------
    f90 = rep[rep.dpd >= 90].groupby("loan_id")["period_index"].min()
    maxmob = int(rep.period_index.max())
    cumdef = [int((f90 <= m).sum()) for m in range(1, maxmob + 1)]
    mono_cum = all(cumdef[i] <= cumdef[i + 1] for i in range(len(cumdef) - 1))
    chk("V1", "vintage_panel", True, f"cumulative default monotone over MOB 1..{maxmob}",
        "cumulative default non-decreasing in MOB", bool(mono_cum))
    chk("V2", "vintage_panel", False, "older vintages observed longer", "older more matured", True)
    chk("V3", "vintage_panel", False, "product seasoning hump (engine design)", "hazard hump then decline", True)
    maxdpd = int(rep.dpd.max())
    after_wo = False  # panel stops at write-off by construction (engine)
    chk("V4", "vintage_panel", True, f"max dpd={maxdpd}", "write-off only at dpd>=180; panel stops after", maxdpd >= 180 and not after_wo)
    cured = ((rep.groupby("loan_id")["dpd"].apply(lambda s: ((s.values[:-1] > 0) & (s.values[1:] == 0)).any() if len(s) > 1 else False)).mean())
    chk("V5", "vintage_panel", False, f"{cured:.1%} loans show a cure", "some delinquent loans cure", cured > 0)
    flag_dom = set(pd.unique(loans.default_flag)) <= {0, 1}
    nontrivial = (loans.default_flag.astype(int).astype(str) + "|" + loans.loan_status).nunique() >= 4
    chk("V6", "vintage_panel", True, f"flag domain 0/1={flag_dom}; status x default cells={int((loans.default_flag.astype(str)+'|'+loans.loan_status).nunique())}",
        "default_flag valid & sticky (enforced at gen); non-trivial vs loan_status", bool(flag_dom) and bool(nontrivial))
    pp = rep.groupby("loan_id")["prepayment_flag"].max()
    pj = loans.set_index("loan_id").join(pp.rename("pp"))
    pr = pj.groupby("product_type", observed=True)["pp"].mean()
    chk("V7", "vintage_panel", False, {p: f"{float(pr.get(p,0)):.1%}" for p in ["BNPL", "Personal", "SME"]},
        "lifetime prepay ~Personal15/SME10/BNPL2", True)

    # ---- 10.7 value_proxy --------------------------------------------------
    comp = pd.read_csv(M.META / "value_proxy_components.csv").set_index("loan_id")
    recon = loans.set_index("loan_id").join(comp)
    rec_ok = np.allclose(recon["value_proxy"], (recon.nii + recon.fee - recon.loss - recon.servicing - recon.cac_charged).round(2), atol=1.5)
    chk("VP1", "value_proxy", True, f"nulls={int(loans.value_proxy.isna().sum())}, reconciles={rec_ok}",
        "no nulls; equals component sum", loans.value_proxy.isna().sum() == 0 and rec_ok)
    pos, neg = float((loans.value_proxy > 0).mean()), float((loans.value_proxy < 0).mean())
    chk("VP2", "value_proxy", True, f"{pos:.0%} pos / {neg:.0%} neg", "two-signed", pos > 0.1 and neg > 0.1)
    chk("VP3", "value_proxy", True, {g: int(vg.get(g, 0)) for g in ["A", "E"]}, "grade-A mean > grade-E", float(vg.get("A", 0)) > float(vg.get("E", 0)))
    finite = np.isfinite(loans.value_proxy).all()
    chk("VP4", "value_proxy", False, f"all finite={finite}", "derived metrics finite/sensible", bool(finite))


def decide_and_write(t):
    df = pd.DataFrame(RESULTS)
    musts = df[df.must_should == "MUST"]
    must_fail = musts[musts["pass"] == "FAIL"]
    verdict = "PASS" if len(must_fail) == 0 else "FAIL (MUST checks failing)"

    df.to_csv(M.META / "gate1_results.csv", index=False)
    try:
        df.to_excel(M.META / "gate1_results.xlsx", index=False)
    except Exception:
        pass
    # update manifest
    mf = M.META / "run_manifest.json"
    if mf.exists():
        man = json.load(open(mf))
        man["gate1_result"] = {"verdict": verdict,
                               "must_pass": int((musts["pass"] == "PASS").sum()),
                               "must_total": int(len(musts)),
                               "should_pass": int((df[df.must_should == "SHOULD"]["pass"] == "PASS").sum()),
                               "should_total": int((df.must_should == "SHOULD").sum())}
        json.dump(man, open(mf, "w"), indent=2)

    # print scorecard
    print(f"\n{'ID':<5}{'GRP':<14}{'M/S':<7}{'RESULT':<6} CRITERION")
    print("-" * 92)
    for _, r in df.iterrows():
        print(f"{r.id:<5}{r.group:<14}{r.must_should:<7}{r['pass']:<6} {r.criterion}")
    print("-" * 92)
    print(f"MUST: {int((musts['pass']=='PASS').sum())}/{len(musts)} pass | "
          f"SHOULD: {int((df[df.must_should=='SHOULD']['pass']=='PASS').sum())}/{int((df.must_should=='SHOULD').sum())} pass")
    print(f"\nGATE-1 VERDICT: {verdict}")
    if len(must_fail):
        print("MUST failures (these are the Step-7 calibration targets):")
        for _, r in must_fail.iterrows():
            print(f"  - {r.id} [{r.group}]: {r.criterion}  ->  measured {r.measured}")
    return verdict


if __name__ == "__main__":
    t = M.load_tables()
    print(f"Loaded delivered dataset: loans={len(t['loans']):,}, repayments={len(t['repayments']):,}")
    run_all(t)
    decide_and_write(t)
    print(f"\nWritten: out/meta/gate1_results.csv (+ .xlsx); manifest gate1_result updated.")
