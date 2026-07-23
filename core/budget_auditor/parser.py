"""
IT-budget form parser.

Ground truth about the real files (verified against 5 actual subsidiary forms:
Subsidiary A, Subsidiary B, Subsidiary B, "sample form A", "sample form B"):

- Metadata block sits in rows 2-6, column C = label, column D = value
  (Год кампании, Версия, Компания, Валюта, Ед. измерения).
- Each sheet (OPEX / CAPEX, and per-branch sheets like ГО/АФ/КазНИПИ) has ONE
  header row, found by locating the cell "Статья бюджета" in column C.
- Below the header, rows come in 3 kinds:
    1) CODE rows      -> col C has a code like "IT-OP-01", col D has the статья name.
                          These only appear once per section; must be forward-filled.
    2) CATEGORY rows  -> col G ("Виды затрат") filled, col H ("Подвид затрат") empty,
                          no numeric plan value. These are template group labels with
                          no data of their own.
    3) LEAF rows      -> a numeric value in a plan/fact/forecast column AND a label
                          in category or subcategory. These are the actual budget
                          line items and the only rows that matter for arithmetic.
- Column SET and column NAMING both drift between subsidiaries/versions of the same
  template -- this is the actual root cause of "some files don't parse right", not
  a structural bug in any one file:
    * "2025 год факт" (Subsidiary A) vs "ФАКТ за 2025 год" (another) --
      same meaning, word order flipped. Must match on tokens ("факт" + a year),
      not a fixed phrase.
    * "2026 год утвержденный план" (Subsidiary A) vs "ПЛАН за 2026 год" (no "утвержденный" at
      all, in newer forms). Same trap: match on tokens ("план" + a year), not phrase.
    * "Описание" and "Обоснование" are two separate columns in Subsidiary A's form, but a
      single merged column "ОПИСАНИЕ и ОБОСНОВАНИЕ" in the newer forms. A header
      that matches more than one signature must be tagged with ALL matching roles,
      not just the first (the previous version had a `break` here that silently
      dropped "justification" for any file using the merged header -- meaning it
      reported "no Обоснование column" for files that actually had one).
    * Some sheets carry TWO "План за <year> год"-style columns in the same row
      (e.g. one for the current campaign year 2027, one left over for 2026).
      Picking "whichever plan column comes first" is wrong -- you have to pick the
      one whose year matches the campaign year from the metadata block, or you
      silently read last year's already-committed plan as if it were this year's.
- A naive `pd.read_excel(sheet, header=0)` breaks on all of this because there is
  no single flat header row with one row per record, and the header text itself
  is not stable across subsidiaries/versions.

Per-year data model
--------------------
Instead of fixed roles ("prior_actual", "prior_plan", "plan_year"), every leaf row
carries a dynamic set of columns named:
    plan_<year>      -- planned amount for <year>       (e.g. plan_2027)
    fact_<year>       -- actual/factual amount for <year> (e.g. fact_2025)
    forecast_<year>   -- forward-looking forecast for <year>
    plan_<year>_q<N>  -- quarterly breakdown of plan_<year>
This makes multi-year comparisons (previous years AND future forecast years, not
just "this year vs. one baseline") a matter of listing whichever plan_*/fact_*/
forecast_* columns exist on a given sheet, which varies by subsidiary.
"""

import re
import openpyxl
import pandas as pd


# Roles that are NOT year-numbered -- looked up by a fixed substring signature.
# ("описание"/"обоснование" match independently so a merged header gets both.)
COLUMN_SIGNATURES = {
    "statya_code": "статья бюджета",
    "category": "виды затрат",
    "subcategory": "подвид затрат",
    "project_flag": "признак проекта",
    "contract": "договор",
    "counterparty": "контрагент",
    "description": "описание",
    "justification": "обоснование",
}


def _norm(s):
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def read_metadata(ws):
    meta = {}
    for r in range(1, 8):
        label = ws.cell(row=r, column=3).value
        value = ws.cell(row=r, column=4).value
        if label:
            meta[_norm(label)] = value
    return {
        "campaign_year": meta.get("год кампании"),
        "version": meta.get("версия"),
        "company": meta.get("компания"),
        "currency": meta.get("валюта"),
        "unit": meta.get("ед. измерения"),
    }


def _find_header_row(ws, max_scan=25):
    for r in range(1, max_scan):
        v = ws.cell(row=r, column=3).value
        if v and _norm(v) == "статья бюджета":
            return r
    return None


def _classify_columns(ws, header_row, max_col):
    """Map column index -> role(s). A column can satisfy more than one fixed
    role (merged "ОПИСАНИЕ и ОБОСНОВАНИЕ" headers) and/or a year-tagged role
    (plan/fact/forecast/quarter) at the same time -- checked independently,
    not as a first-match-wins chain."""
    colmap = {}       # role -> col index (fixed, non-year roles)
    year_cols = {}    # col index -> ("plan"|"fact"|"forecast"|"quarter", year[, quarter])

    for c in range(3, max_col + 1):
        label = _norm(ws.cell(row=header_row, column=c).value)
        if not label:
            continue

        for role, sig in COLUMN_SIGNATURES.items():
            if sig in label:
                colmap[role] = c
        if any(sig in label for sig in COLUMN_SIGNATURES.values()):
            continue  # a fixed-role header is never also a year column

        m = re.search(r"(\d)\s*кв\D*(\d{4})", label)
        if m:
            year_cols[c] = ("quarter", int(m.group(2)), int(m.group(1)))
            continue

        year_m = re.search(r"(20\d{2})", label)
        if not year_m:
            continue
        year = int(year_m.group(1))
        if "прогноз" in label:
            year_cols[c] = ("forecast", year)
        elif "факт" in label:
            year_cols[c] = ("fact", year)
        elif "план" in label:
            year_cols[c] = ("plan", year)

    return colmap, year_cols


def parse_sheet(ws, sheet_name, campaign_year=None):
    header_row = _find_header_row(ws)
    if header_row is None:
        return None, {"sheet": sheet_name, "error": "no_header_found"}

    max_col = ws.max_column
    colmap, year_cols = _classify_columns(ws, header_row, max_col)

    # The "current" plan column is whichever plan_<year> column matches the
    # campaign year from the metadata block -- NOT "whichever comes first".
    # Some sheets legitimately carry two plan-year columns at once (this
    # year's target + a leftover column for the year before).
    plan_year_cols = [c for c, v in year_cols.items() if v[0] == "plan"]
    plan_year_col = None
    if campaign_year:
        plan_year_col = next((c for c in plan_year_cols if year_cols[c][1] == campaign_year), None)
    if plan_year_col is None and plan_year_cols:
        # No campaign-year metadata, or none of the plan columns match it --
        # fall back to the latest year found rather than the leftmost column.
        plan_year_col = max(plan_year_cols, key=lambda c: year_cols[c][1])
    plan_year = year_cols[plan_year_col][1] if plan_year_col else campaign_year

    rows = []
    cur_code, cur_code_name = None, None
    n_blank = n_category = n_leaf = 0

    for r in range(header_row + 1, ws.max_row + 1):
        code = ws.cell(row=r, column=3).value
        code_name = ws.cell(row=r, column=4).value
        category = ws.cell(row=r, column=7).value
        subcat = ws.cell(row=r, column=8).value
        plan_val = ws.cell(row=r, column=plan_year_col).value if plan_year_col else None

        if code:
            # Every form ends with a grand-total block whose code cell reads
            # something like "Общий результат" (not a real Статья бюджета
            # code) followed by a row that repeats the sum in the same plan
            # column. If we don't stop here, that footer sum gets ingested
            # as if it were one more line item and silently doubles the total.
            if re.search(r"итог|результат", _norm(code)):
                break
            cur_code, cur_code_name = code, code_name
            continue

        is_leaf = isinstance(plan_val, (int, float)) and bool(category or subcat)
        if not is_leaf:
            if category and not subcat:
                n_category += 1
            else:
                n_blank += 1
            continue

        n_leaf += 1
        rec = {
            "sheet": sheet_name,
            "statya_code": cur_code,
            "statya_name": cur_code_name,
            "category": category,
            "subcategory": subcat,
            "item_name": subcat or category,
        }
        for role, c in colmap.items():
            if role in ("statya_code", "category", "subcategory"):
                continue
            rec[role] = ws.cell(row=r, column=c).value
        for c, spec in year_cols.items():
            v = ws.cell(row=r, column=c).value
            kind, year = spec[0], spec[1]
            if kind == "quarter":
                rec[f"plan_{year}_q{spec[2]}"] = v
            else:
                rec[f"{kind}_{year}"] = v
        rows.append(rec)

    df = pd.DataFrame(rows)
    years_present = sorted({v[1] for v in year_cols.values()})
    stats = {
        "sheet": sheet_name,
        "header_row": header_row,
        "n_leaf_rows": n_leaf,
        "n_category_rows": n_category,
        "n_blank_template_rows": n_blank,
        "has_description": "description" in colmap,
        "has_justification": "justification" in colmap,
        "plan_year": plan_year,
        "years_present": years_present,
    }
    return df, stats


def parse_workbook(path):
    wb = openpyxl.load_workbook(path, data_only=True)
    sheets_out = {}
    meta = None
    for sn in wb.sheetnames:
        ws = wb[sn]
        if meta is None:
            m = read_metadata(ws)
            if m.get("company") or m.get("campaign_year"):
                meta = m
    if meta is None:
        meta = {}
    campaign_year = meta.get("campaign_year")

    for sn in wb.sheetnames:
        ws = wb[sn]
        df, stats = parse_sheet(ws, sn, campaign_year=campaign_year)
        sheets_out[sn] = {"df": df, "stats": stats}
    return meta, sheets_out


def aggregate_workbook(sheets_out):
    """Combine every parsed sheet into one workbook-wide dataframe, tagged
    with OPEX/CAPEX and branch (ГО/АФ/КазНИПИ/...). Needed for: the annual
    CAPEX-vs-OPEX comparison table, the multi-year trend view, and a chatbot
    that can answer questions spanning the whole file instead of just
    whatever sheet happens to be selected in a dropdown.

    Classification rule (confirmed against the real files): any sheet whose
    name contains "CAPEX" is a CAPEX sheet; everything else (OPEX, ГО, АФ,
    КазНИПИ, ...) is OPEX. Branch = the sheet name with "CAPEX" stripped
    (so "ГО" and "ГО CAPEX" both roll up to branch "ГО"); a bare "OPEX" or
    "CAPEX" sheet with no branch prefix rolls up to branch "(головная)".
    """
    frames = []
    for sn, d in sheets_out.items():
        df = d["df"]
        if df is None or df.empty:
            continue
        is_capex = "capex" in _norm(sn)
        branch = re.sub(r"(?i)capex", "", sn).strip() or "(головная)"
        if branch.upper() == "OPEX":
            branch = "(головная)"
        f = df.copy()
        f["cost_type"] = "CAPEX" if is_capex else "OPEX"
        f["branch"] = branch
        f["plan_year"] = d["stats"].get("plan_year")
        frames.append(f)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def available_years(combined_df):
    """Every year for which at least one plan_/fact_/forecast_ column exists
    anywhere in the combined workbook, in chronological order, tagged with
    which kind of value it is (a year can have more than one kind, e.g. a
    fact AND a plan for the same recent year)."""
    years = {}
    for col in combined_df.columns:
        m = re.match(r"(plan|fact|forecast)_(\d{4})$", col)
        if m:
            years.setdefault(int(m.group(2)), set()).add(m.group(1))
    return dict(sorted(years.items()))


def year_trend(combined_df):
    """Tidy long-format table: one row per (year, kind), summed across the
    whole workbook -- this is the 'compare with other years (previous or
    forecast)' view. kind is 'план' / 'факт' / 'прогноз' in Russian for
    direct display."""
    years = available_years(combined_df)
    kind_label = {"plan": "план", "fact": "факт", "forecast": "прогноз"}
    rows = []
    for year, kinds in years.items():
        for kind in kinds:
            col = f"{kind}_{year}"
            if col in combined_df.columns:
                total = pd.to_numeric(combined_df[col], errors="coerce").sum()
                rows.append({"Год": year, "Тип": kind_label[kind], "Сумма": total})
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(["Год", "Тип"]).reset_index(drop=True)

    # Sanity check: every year in this file should be roughly the same order
    # of magnitude (it's the same budget, just different years). If one
    # year's total is off by >20x *above* the median of all years, that's
    # much more likely a units mistake in the source file (e.g. someone
    # entered a факт column in raw tenge while every план column is in
    # тыс.) than a genuine 20x YoY swing -- flag it instead of silently
    # plotting it. A year at/near zero is a different, more mundane thing
    # (a forecast column the subsidiary just never filled in) and is not
    # flagged as a units problem.
    median = out["Сумма"].replace(0, pd.NA).median()
    if pd.notna(median) and median != 0:
        ratio = out["Сумма"] / median
        out["unit_mismatch_suspected"] = ratio > 20
    else:
        out["unit_mismatch_suspected"] = False
    return out


def capex_opex_table(combined_df):
    """Small annual CAPEX-vs-OPEX comparison table: one row per cost type,
    total plan for the campaign year, share of combined total, and YoY delta
    against the best available prior-year baseline (fact for campaign_year-1
    if present, else plan for campaign_year-1)."""
    if combined_df is None or combined_df.empty:
        return pd.DataFrame()

    plan_year = combined_df["plan_year"].dropna().iloc[0] if combined_df["plan_year"].notna().any() else None
    plan_col = f"plan_{plan_year}" if plan_year else None
    baseline_col = _best_baseline_col(combined_df, plan_year)
    grand_total = pd.to_numeric(combined_df[plan_col], errors="coerce").sum() if plan_col in combined_df.columns else None

    rows = []
    for cost_type, g in combined_df.groupby("cost_type"):
        total = pd.to_numeric(g[plan_col], errors="coerce").sum() if plan_col in g.columns else None
        row = {"Тип": cost_type, f"План {plan_year}": total,
               "Доля": (total / grand_total * 100) if grand_total else None}
        if baseline_col and baseline_col in g.columns:
            base_total = pd.to_numeric(g[baseline_col], errors="coerce").sum()
            row["База пред. года"] = base_total
            row["Δ %"] = ((total - base_total) / base_total * 100) if base_total else None
        rows.append(row)
    return pd.DataFrame(rows)


def _best_baseline_col(df, plan_year):
    """Prefer last year's FACT over last year's PLAN as the YoY baseline --
    an actual is a better comparison point than a plan that may not have
    been spent in full. Falls back to None if neither exists on this sheet
    (structurally absent, e.g. Subsidiary B forms)."""
    if not plan_year:
        return None
    prev = plan_year - 1
    if f"fact_{prev}" in df.columns and df[f"fact_{prev}"].notna().any():
        return f"fact_{prev}"
    if f"plan_{prev}" in df.columns and df[f"plan_{prev}"].notna().any():
        return f"plan_{prev}"
    return None


def _flag_quarterly(df, plan_year, plan_col):
    """Quarterly reconciliation, mirroring budget_audit.quarterly_reconciliation
    but on the dynamic per-year names (plan_<year>_q1..q4 vs plan_<year>):
    |annual - Σquarters| / annual over 1% flags the row; over 5% = "critical",
    otherwise "minor". Rows missing the annual value or any quarter, and rows
    with a non-positive annual plan, are skipped — never flagged. Sheets whose
    form has no quarterly breakdown at all keep every flag False (a structural
    gap, same treatment as a missing prior-year baseline)."""
    df["quarterly_sum"] = pd.NA
    df["quarterly_gap_abs"] = pd.NA
    df["quarterly_gap_pct"] = pd.NA
    df["quarterly_severity"] = pd.NA
    df["quarterly_mismatch_flag"] = False

    quarter_cols = [f"plan_{plan_year}_q{n}" for n in range(1, 5)] if plan_year else []
    if not quarter_cols or plan_col not in df.columns \
            or not all(c in df.columns for c in quarter_cols):
        return df

    annual = pd.to_numeric(df[plan_col], errors="coerce")
    quarters = df[quarter_cols].apply(pd.to_numeric, errors="coerce")
    complete = annual.notna() & quarters.notna().all(axis=1) & (annual > 0)
    qsum = quarters.sum(axis=1)
    gap = annual - qsum
    gap_pct = gap.abs() / annual * 100

    df.loc[complete, "quarterly_sum"] = qsum[complete]
    df.loc[complete, "quarterly_gap_abs"] = gap[complete]
    df.loc[complete, "quarterly_gap_pct"] = gap_pct[complete]
    flagged = complete & (gap_pct > 1.0)
    df.loc[flagged, "quarterly_severity"] = gap_pct[flagged].map(
        lambda pct: "critical" if pct > 5 else "minor"
    )
    df["quarterly_mismatch_flag"] = flagged
    return df


def flag_rows(df, stats, threshold_pct=15.0):
    """Return df with added risk/quality flag columns. Works even when
    prior-year columns don't exist in this subsidiary's form."""
    if df is None or df.empty:
        return df

    df = df.copy()
    plan_year = stats.get("plan_year")
    plan_col = f"plan_{plan_year}" if plan_year else None
    baseline_col = _best_baseline_col(df, plan_year)

    if baseline_col and plan_col and plan_col in df.columns:
        base = pd.to_numeric(df[baseline_col], errors="coerce")
        plan = pd.to_numeric(df[plan_col], errors="coerce")
        df["delta_abs"] = plan - base
        df["delta_pct"] = (plan - base) / base.replace(0, pd.NA) * 100
        df["high_change_flag"] = df["delta_pct"].abs() > threshold_pct
    else:
        df["delta_abs"] = pd.NA
        df["delta_pct"] = pd.NA
        df["high_change_flag"] = False

    if not stats["has_justification"]:
        df["missing_justification_flag"] = True
    else:
        df["missing_justification_flag"] = df.get(
            "justification", pd.Series(dtype=object)
        ).isna()

    if plan_col and plan_col in df.columns:
        plan_numeric = pd.to_numeric(df[plan_col], errors="coerce")
        cutoff = plan_numeric.quantile(0.90) if plan_numeric.notna().any() else None
        df["high_price_flag"] = plan_numeric >= cutoff if cutoff is not None else False
    else:
        df["high_price_flag"] = False

    return _flag_quarterly(df, plan_year, plan_col)
