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

# Budget-code format confirmed against real files (e.g. "IT-CA-01"): 2-4
# uppercase letters, a 2-letter subsection, a 2-digit sequence number, all
# hyphen-separated. Checked case-sensitively against the raw cell text (not
# _norm(), which lowercases) -- codes are always stored uppercase in the
# source files.
_CODE_RE = re.compile(r"^[A-Z]{2,4}-[A-Z]{2}-\d{2}$")

# Sheet-terminating footer marker. Confirmed against 4 real files: the ONE
# reliable one-time sheet-ending label is "Общий результат" in the Статья
# бюджета (code) column -- checked here ONLY, as a substring search, and
# ONLY against that column. An earlier version searched this same pattern
# across every column (including free-text description/justification), which
# false-positived on ordinary prose like "результатов инвентаризации
# оборудования" in a justification sentence and truncated the sheet after
# 1-2 rows -- never search this against free-text columns again.
_SHEET_END_RE = re.compile(r"итог|результат")

# A bare per-budget-code subtotal row: category/subcategory holds EXACTLY
# (full match, not substring) "результат" or "итого" after normalizing
# whitespace/case. Real files repeat this after EVERY budget-code section
# (IT-OP-01, IT-OP-02, IT-OP-03, ...) -- it is a single-row skip, not a
# sheet-ending marker, and (being a full-match check) never fires on free
# text that merely contains "результат" as a substring.
_SECTION_SUBTOTAL_LABELS = frozenset({"результат", "итого"})


def _norm(s):
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _is_valid_budget_code(code) -> bool:
    if code is None:
        return False
    return bool(_CODE_RE.match(str(code).strip()))


def _is_sheet_end_row(ws, row) -> bool:
    """True if the Статья бюджета (code) column carries the sheet's one-time
    ending label (e.g. "Общий результат"). Checked ONLY against this column
    -- see _SHEET_END_RE."""
    v = ws.cell(row=row, column=3).value
    return bool(v and _SHEET_END_RE.search(_norm(v)))


def _is_section_subtotal_row(category, subcat) -> bool:
    """True if category or subcategory is EXACTLY "результат"/"итого" -- a
    per-budget-code subtotal that repeats after every section. Excluded from
    leaf ingestion, but never terminates the sheet -- see
    _SECTION_SUBTOTAL_LABELS."""
    return _norm(category) in _SECTION_SUBTOTAL_LABELS or _norm(subcat) in _SECTION_SUBTOTAL_LABELS


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
    cur_category, cur_subcat = None, None
    n_blank = n_category = n_leaf = 0

    for r in range(header_row + 1, ws.max_row + 1):
        # Every form ends with a grand-total block, e.g. "Общий результат" in
        # the code column -- the one true one-time sheet-ending label
        # confirmed across every real file examined. Checked ONLY against
        # the code column (never free text) so this can't false-positive on
        # ordinary prose elsewhere in the row -- see _is_sheet_end_row.
        if _is_sheet_end_row(ws, r):
            break

        code = ws.cell(row=r, column=3).value
        code_name = ws.cell(row=r, column=4).value
        category = ws.cell(row=r, column=7).value
        subcat = ws.cell(row=r, column=8).value
        plan_val = ws.cell(row=r, column=plan_year_col).value if plan_year_col else None

        # A code cell only counts as a genuine section declaration if it
        # looks like a real code, OR the row has no category/subcategory of
        # its own (the shape of a real declaration row in every file
        # examined: col C + col D only, nothing else). A real file was seen
        # with the PREPARER'S NAME in this column ("НЕКОДОВОЕ_ЗНАЧЕНИЕ" -- not a code,
        # not empty) on rows that also carried real category/item text;
        # treating that as a new declaration silently discarded the row's
        # own data. Deliberately NOT keyed to plan_val/the campaign-year
        # column specifically: a row can have real category/item data but no
        # *current-year* number (e.g. only a fact_2025 or plan_2026 value) --
        # it's still a data row, not a declaration, even though this year's
        # plan is empty for it. When neither holds, cur_code is left as-is
        # (the row still belongs to whatever section is already open) and
        # the row falls through to be evaluated as a leaf below.
        if code and (_is_valid_budget_code(code) or not (category or subcat)):
            cur_code, cur_code_name = code, code_name
            cur_category, cur_subcat = None, None
            continue

        # A bare per-section subtotal ("Результат"/"Итого" alone in category
        # or subcategory) repeats after every budget-code section in real
        # files -- exclude just this row, keep reading the sections after it.
        if _is_section_subtotal_row(category, subcat):
            cur_category, cur_subcat = None, None
            n_blank += 1
            continue

        # Real files also split one leaf item's category label across
        # several rows: a "group header" row carries category (and often its
        # own item+value), then continuation rows repeat only the value with
        # category/subcategory left blank. Forward-fill within the open
        # section so continuation rows are still recognized as leaf rows
        # instead of being silently dropped for lacking a category of their
        # own -- a no-op for forms where every leaf row already carries its
        # own category/subcategory (e.g. Subsidiary A-style forms).
        if category:
            cur_category = category
        if subcat:
            cur_subcat = subcat
        eff_category = category or cur_category
        eff_subcat = subcat or cur_subcat

        is_leaf = isinstance(plan_val, (int, float)) and bool(eff_category or eff_subcat)
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
            "category": eff_category,
            "subcategory": eff_subcat,
            "item_name": eff_subcat or eff_category,
            # A code that doesn't match the "AA-AA-00"-style format is a
            # data-quality warning, not a parse failure -- surfaced downstream
            # (workbook_summary_text, the flagged-статьи UI), never fatal.
            "anomalous_code_flag": not _is_valid_budget_code(cur_code),
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
        "n_anomalous_code_rows": int(df["anomalous_code_flag"].sum()) if "anomalous_code_flag" in df.columns else 0,
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

    # pandas deprecated (and, as of 3.0, finalized) letting concat's dtype
    # inference decide the result dtype of a column that's genuinely PRESENT
    # but entirely pd.NA (object dtype) on one sheet while holding real
    # numbers on another -- e.g. delta_abs is a real int column on an
    # Subsidiary A-style OPEX sheet but a bare `pd.NA` broadcast (see flag_rows() /
    # _flag_quarterly()) on a Subsidiary B-style CAPEX sheet with no baseline year.
    # Concatenating that mix relies on concat to pick the union dtype (today:
    # object, since int + object always promotes to object) -- exactly the
    # inference the warning is about. Make the dtype explicit instead of
    # inferred: cast every frame's copy of such a column to object before
    # concatenating, which is the dtype the mix already resolves to, so the
    # result is unchanged but no longer depends on concat's own inference.
    # A column entirely ABSENT from a sheet (e.g. fact_2025 on a sheet with no
    # факт column at all) is a different, ordinary reindex-and-fill case, not
    # this deprecation, and is deliberately left untouched.
    all_columns = list(dict.fromkeys(c for f in frames for c in f.columns))
    unstable_cols = {
        c for c in all_columns
        if any(c in f.columns and f[c].dtype == object and f[c].isna().all() for f in frames)
        and any(c in f.columns and not f[c].isna().all() for f in frames)
    }
    if unstable_cols:
        aligned = []
        for f in frames:
            cols_here = unstable_cols & set(f.columns)
            if cols_here:
                f = f.copy()
                for c in cols_here:
                    f[c] = f[c].astype(object)
            aligned.append(f)
        frames = aligned

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
    gap, same treatment as a missing prior-year baseline).

    Some subsidiaries report quarters нарастающим итогом (cumulative -- Q4
    already IS the annual total, Q3 already includes Q1+Q2, etc.) rather than
    discrete per-quarter amounts. Summing four cumulative quarters against the
    annual figure double/triple/quadruple-counts and flags every single row
    "critical" — a 100% false positive on files using that convention. A row
    is treated as cumulative when Q4 is within ~1% of the annual total AND
    Q1<=Q2<=Q3<=Q4 (non-decreasing — a flat quarter with zero activity is
    still a valid cumulative series, so this doesn't require strict growth);
    in that case Q4 is compared directly to the annual figure instead of the
    four-quarter sum. Otherwise the existing discrete sum-and-compare holds."""
    df["quarterly_sum"] = pd.NA
    df["quarterly_gap_abs"] = pd.NA
    df["quarterly_gap_pct"] = pd.NA
    df["quarterly_severity"] = pd.NA
    df["quarterly_mismatch_flag"] = False
    df["quarterly_cumulative_detected"] = False

    quarter_cols = [f"plan_{plan_year}_q{n}" for n in range(1, 5)] if plan_year else []
    if not quarter_cols or plan_col not in df.columns \
            or not all(c in df.columns for c in quarter_cols):
        return df

    annual = pd.to_numeric(df[plan_col], errors="coerce")
    quarters = df[quarter_cols].apply(pd.to_numeric, errors="coerce")
    complete = annual.notna() & quarters.notna().all(axis=1) & (annual > 0)
    q1, q2, q3, q4 = (quarters[c] for c in quarter_cols)

    discrete_sum = quarters.sum(axis=1)
    q4_gap_pct = (annual - q4).abs() / annual.replace(0, pd.NA) * 100
    monotonic = (q1 <= q2) & (q2 <= q3) & (q3 <= q4)
    is_cumulative = complete & monotonic & (q4_gap_pct <= 1.0)

    effective_sum = discrete_sum.where(~is_cumulative, q4)
    gap = annual - effective_sum
    gap_pct = gap.abs() / annual * 100

    df.loc[complete, "quarterly_sum"] = effective_sum[complete]
    df.loc[complete, "quarterly_gap_abs"] = gap[complete]
    df.loc[complete, "quarterly_gap_pct"] = gap_pct[complete]
    df.loc[complete, "quarterly_cumulative_detected"] = is_cumulative[complete]
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
