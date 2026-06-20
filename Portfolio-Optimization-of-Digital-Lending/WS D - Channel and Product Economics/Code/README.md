# Workstream D — EDA reproducibility package

Regenerates the entire Workstream D EDA (the workbook and the seven chart
exhibits) from the **frozen Workstream C dataset**. Provided for audit trail and
reproducibility — it is *not* needed to start Workstream E (E reads the EDA
outputs), but it lets a reviewer re-derive every number from source.

## Contents
- `wsD_analysis.py` — the single, self-contained generating script.
- `requirements.txt` — Python dependencies.
- `README.md` — this file.

## What it reproduces
**Ours (recomputed from source):**
- Q3 economics — Slice A (product), Slice B (product × tenure), Slice C (product × ticket)
- Equal-MOB default check (controls for loan age)
- Vintage / cohort analysis (raw vs equal-age, grade-mix drift)
- Roll-rate / delinquency-bucket migration + time-to-first-delinquency
- Portfolio overview (KPIs + product concentration)
- Product × channel crossover (the capstone)
- All seven PNG exhibits

**Coworker's (incorporated, not recomputed):** Q2 channel economics. If the Q2
controlled workbook is supplied via `--q2`, its channel table is folded in and a
reconciliation + crossover sheet are added. Q2 numbers are *sourced from that
file*, never regenerated here. (The coworker's verbatim DSA-decomposition and
findings sheets live in their own workbook; this script does not re-transcribe
them.)

## Inputs (frozen WS-C output — seed 20260603, Gate 1 PASS 32/32 MUST)
```
<DATA>/data/parquet/loans.parquet
<DATA>/data/parquet/customers.parquet
<DATA>/data/parquet/repayments.parquet
<DATA>/meta/value_proxy_components.csv
[optional] wsD_Q2_controlled_workbook.xlsx
```

## Run
```bash
pip install -r requirements.txt
python wsD_analysis.py --data /path/to/wsC/out --out ./wsD_outputs --q2 /path/to/wsD_Q2_controlled_workbook.xlsx
```
`--q2` is optional. Without it you get `Q3_results_log.xlsx` (10 sheets) + charts;
with it you also get `WorkstreamD_consolidated_log.xlsx` (13 sheets).

## Outputs (in `--out`)
- `Q3_results_log.xlsx`, `WorkstreamD_consolidated_log.xlsx`
- `sliceA_product_bar.png`, `sliceB_tenure_heatmap.png`, `sliceC_ticket_heatmap.png`
- `fragment1_vintage.png`, `fragment2_rollrate.png`, `fragment3_portfolio_overview.png`, `fragment4_product_channel.png`

## Definitions (must match Q2)
- `value_proxy` (profit) = NII + Fee − Loss − Servicing − CAC  *(already net of CAC)*
- `margin_before_cac` = NII + Fee − Loss − Servicing
- `default` = ever 90+ DPD; `maturity` = mean months-on-book ÷ mean tenure

## Built-in checks
- Asserts no duplicate `loan_id` and a clean one-to-one merge.
- **Asserts `value_proxy` reconciles to its components to < 1 rupee** (one number, one source). The script stops if it doesn't.
- Ticket bands use documented ranges with an **auto-fallback to data thirds** if any band is degenerate, and print a `[guard]` line if they do.

## Notes for the auditor
- Deterministic: the data is frozen, so re-running reproduces identical numbers.
- Workbook derived columns are **live Excel formulas**; open in Excel/LibreOffice (or run a recalc) to populate cached values.
- Single M24 snapshot with mixed loan ages → low-maturity cells show profit/default *to date*, not lifetime (long/large cells are understated). The `maturity` column flags this per cell.
- Lifetime PD projection is deliberately *out of scope* (it belongs to the optimization workstream); everything here is realised, observed-to-date.
