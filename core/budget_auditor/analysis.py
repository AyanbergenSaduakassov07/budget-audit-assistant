"""
Text summary generation + a small rule-based query engine over one parsed
budget sheet. No LLM call is wired in here (no API key is available in this
environment) -- `answer_query()` is deliberately written as a single function
so it's a one-line swap to route unmatched questions to an LLM later
(see the docstring at the bottom).
"""

import re
import pandas as pd

from .parser import _best_baseline_col, year_trend


def normalize_unit(unit):
    """Some forms store the unit as text ('тыс.тенге'), others as a bare
    number (1000) with no currency word at all -- another small structural
    inconsistency between subsidiaries. Normalize both to a readable label."""
    if unit is None:
        return ""
    if isinstance(unit, (int, float)):
        return "тыс." if unit == 1000 else f"x{unit}"
    return str(unit)


def format_money(v, unit_label=""):
    if v is None or pd.isna(v):
        return "н/д"
    unit_label = normalize_unit(unit_label)
    return f"{v:,.0f} {unit_label}".replace(",", " ").strip()


def sheet_summary_text(sheet_name, meta, df, stats):
    """Russian-language paragraph describing one parsed sheet."""
    if df is None or df.empty:
        return (
            f"Лист «{sheet_name}»: строк с реальными данными не найдено. "
            f"Похоже, форма по этому разделу подана пустой (только шаблон, "
            f"без заполненных статей расхода)."
        )

    plan_col = f"plan_{stats['plan_year']}"
    total = pd.to_numeric(df[plan_col], errors="coerce").sum() if plan_col in df.columns else None
    n_items = len(df)
    n_codes = df["statya_code"].nunique()
    unit = meta.get("unit", "")

    parts = [
        f"Лист «{sheet_name}»: {n_items} заполненных статей расходов "
        f"по {n_codes} кодам бюджета, план на {stats['plan_year']} год "
        f"составляет {format_money(total, str(unit))}."
    ]

    base_col = _best_baseline_col(df, stats.get("plan_year"))
    if base_col:
        base_total = pd.to_numeric(df[base_col], errors="coerce").sum()
        base_label = "факта" if base_col.startswith("fact_") else "утверждённого плана"
        if base_total and base_total != 0:
            delta_pct = (total - base_total) / base_total * 100
            direction = "рост" if delta_pct > 0 else "снижение"
            parts.append(
                f"По сравнению с базой прошлого года ({base_label}, {format_money(base_total, str(unit))}) "
                f"это {direction} на {abs(delta_pct):.1f}% "
                f"({format_money(total - base_total, str(unit))})."
            )
    else:
        parts.append(
            "В форме нет данных за прошлый год (ни факта, ни утверждённого плана), "
            "поэтому динамику год-к-году посчитать невозможно -- это структурный "
            "пробел формы, а не ошибка загрузки."
        )

    years = stats.get("years_present") or []
    if len(years) > 1:
        parts.append(
            f"В форме есть данные за {len(years)} год(а/лет): {', '.join(str(y) for y in years)} "
            f"-- доступно сравнение не только с прошлым годом, но и полный тренд по прогнозным годам."
        )

    if not stats["has_justification"]:
        parts.append(
            "В форме полностью отсутствует колонка «Обоснование» -- ни одна статья "
            "не может быть проверена на предмет обоснованности расхода."
        )
    else:
        n_missing_just = int(df["missing_justification_flag"].sum()) if "missing_justification_flag" in df else 0
        if n_missing_just:
            parts.append(f"{n_missing_just} из {n_items} статей не имеют заполненного обоснования.")

    n_high_change = int(df["high_change_flag"].sum()) if "high_change_flag" in df else 0
    if n_high_change:
        parts.append(f"{n_high_change} статей показывают изменение более чем на заданный порог по сравнению с прошлым годом -- рекомендуется проверить вручную.")

    n_high_price = int(df["high_price_flag"].sum()) if "high_price_flag" in df else 0
    if n_high_price:
        parts.append(f"{n_high_price} статей входят в верхние 10% по стоимости на этом листе -- потенциально рискованные по цене позиции.")

    return " ".join(parts)


def workbook_summary_text(meta, sheet_results):
    header = []
    company = meta.get("company") or "Компания не указана в форме"
    header.append(
        f"{company}. Кампания {meta.get('campaign_year', '?')} год, "
        f"версия «{meta.get('version', '?')}», валюта: {meta.get('currency', '?')}, "
        f"единица измерения: {meta.get('unit', '?')}."
    )
    body = []
    for sn, d in sheet_results.items():
        body.append(sheet_summary_text(sn, meta, d["df"], d["stats"]))
    return "\n\n".join(header + body)


def year_trend_text(combined_df):
    """Narrative for the multi-year trend table -- also surfaces the
    unit-mismatch sanity check (e.g. a факт column entered in raw currency
    while план columns are in тыс. produces a >20x outlier year that's
    almost certainly a units mistake, not a real budget swing)."""
    yt = year_trend(combined_df)
    if yt.empty:
        return "Недостаточно годовых колонок для построения тренда."
    suspects = yt[yt["unit_mismatch_suspected"]]
    if suspects.empty:
        return "Тренд по годам выглядит согласованным по порядку величины."
    lines = [f"{r.Год} ({r.Тип}): {r.Сумма:,.0f}".replace(",", " ") for r in suspects.itertuples()]
    return (
        "Внимание: следующие значения отличаются от остальных лет более чем в 20 раз -- "
        "это выглядит как ошибка единиц измерения в исходном файле (например, факт внесён "
        "в тенге, а план -- в тыс. тенге), а не как реальный скачок бюджета:\n" + "\n".join(lines)
    )


_MONTHS = {}  # placeholder for future NL date parsing


def answer_query(question, df, stats, meta):
    """Deterministic keyword/year/regex based Q&A over one parsed sheet.
    Covers: 'бюджет на <year>', 'сколько потратили на <keyword>',
    'сколько статей', 'самые дорогие / крупные', 'какие статьи без обоснования'.
    Returns a plain-text Russian answer, or None if nothing matched (caller
    can then fall back to an LLM call -- see module docstring).
    """
    if df is None or df.empty:
        return "В этом листе нет заполненных данных для анализа."

    q = question.lower().strip()
    unit = str(meta.get("unit", ""))

    year_match = re.search(r"(20\d{2})", q)
    year = int(year_match.group(1)) if year_match else None

    if any(w in q for w in ["без обоснован", "нет обоснован", "отсутств"]) and "обоснован" in q:
        if not stats["has_justification"]:
            return "В форме нет колонки «Обоснование» вообще -- обоснование отсутствует для 100% статей."
        missing = df[df["missing_justification_flag"]]
        if missing.empty:
            return "Все статьи имеют заполненное обоснование."
        names = ", ".join(missing["item_name"].astype(str).head(10))
        return f"{len(missing)} статей без обоснования: {names}{'...' if len(missing) > 10 else ''}."

    if any(w in q for w in ["самые дорог", "самые крупн", "топ", "риск"]):
        plan_col = f"plan_{stats['plan_year']}"
        if plan_col not in df.columns:
            return "Нет данных плана для ранжирования."
        top = df.nlargest(5, plan_col)[["item_name", plan_col]]
        lines = [f"{r.item_name}: {format_money(getattr(r, plan_col), unit)}" for r in top.itertuples()]
        return "Топ-5 самых крупных статей:\n" + "\n".join(lines)

    if any(w in q for w in ["сколько статей", "количество статей", "число статей"]):
        return f"Заполненных статей расхода: {len(df)}."

    if year and (f"plan_{year}" in df.columns):
        total = pd.to_numeric(df[f"plan_{year}"], errors="coerce").sum()
        return f"План на {year} год: {format_money(total, unit)}."

    if year and (f"forecast_{year}" in df.columns):
        total = pd.to_numeric(df[f"forecast_{year}"], errors="coerce").sum()
        return f"Прогноз на {year} год: {format_money(total, unit)}."

    # keyword search across item_name / category / subcategory / counterparty
    tokens = [t for t in re.split(r"[^а-яa-zё0-9]+", q) if len(t) > 3]
    if tokens:
        mask = pd.Series(False, index=df.index)
        for col in ["item_name", "category", "subcategory", "counterparty"]:
            if col in df.columns:
                for t in tokens:
                    mask |= df[col].astype(str).str.lower().str.contains(t, na=False)
        matched = df[mask]
        if not matched.empty:
            plan_col = f"plan_{stats['plan_year']}"
            total = pd.to_numeric(matched[plan_col], errors="coerce").sum() if plan_col in matched.columns else None
            names = ", ".join(matched["item_name"].astype(str).head(5))
            return (
                f"Найдено {len(matched)} совпадающих статей ({names}"
                f"{'...' if len(matched) > 5 else ''}). "
                f"Суммарный план: {format_money(total, unit)}."
            )

    return None  # no rule matched -- plug in an LLM fallback here


# --- Where to add a real LLM later --------------------------------------
# def answer_query_llm(question, df, meta, api_key):
#     context = df.to_csv(index=False)
#     prompt = f"Данные бюджета:\n{context}\n\nВопрос: {question}\nОтветь кратко на русском."
#     # call an LLM here, return response.text


def chat_answer(question, combined_df, meta):
    """Workbook-wide chatbot answer -- unlike answer_query() (one sheet),
    this sees every sheet/branch/cost-type at once, so it can answer things
    like 'сколько CAPEX в 2027', 'какая ветка тратит больше', 'сравни ОРЕХ и
    CAPEX', without the user having to pick a sheet from a dropdown first.
    Deterministic/rule-based -- same reasoning as answer_query(), just over
    the aggregated frame. Returns None if nothing matched (plug in an LLM
    call at that point once you have an API key wired up).
    """
    if combined_df is None or combined_df.empty:
        return "В файле нет заполненных данных для анализа."

    q = question.lower().strip()
    unit = meta.get("unit", "")
    plan_year = combined_df["plan_year"].dropna().iloc[0] if combined_df["plan_year"].notna().any() else None
    plan_col = f"plan_{plan_year}" if plan_year else None

    year_match = re.search(r"(20\d{2})", q)
    year = int(year_match.group(1)) if year_match else None
    year_col = f"plan_{year}" if year and f"plan_{year}" in combined_df.columns else (
        f"forecast_{year}" if year and f"forecast_{year}" in combined_df.columns else None
    )

    wants_capex = "capex" in q or "капекс" in q or "капвлож" in q
    wants_opex = "opex" in q or "опекс" in q or "операционн" in q

    # CAPEX vs OPEX comparison
    if ("сравни" in q or "сравнение" in q) and ("capex" in q or "opex" in q or "капекс" in q or "опекс" in q):
        col = year_col or plan_col
        opex_total = pd.to_numeric(combined_df[combined_df["cost_type"] == "OPEX"][col], errors="coerce").sum()
        capex_total = pd.to_numeric(combined_df[combined_df["cost_type"] == "CAPEX"][col], errors="coerce").sum()
        yr = year or plan_year
        return (f"За {yr} год: OPEX = {format_money(opex_total, unit)}, "
                f"CAPEX = {format_money(capex_total, unit)}. "
                f"Соотношение OPEX/CAPEX ~ {opex_total / capex_total:.1f}:1." if capex_total else
                f"За {yr} год: OPEX = {format_money(opex_total, unit)}, CAPEX данных нет.")

    if wants_capex or wants_opex:
        col = year_col or plan_col
        cost_type = "CAPEX" if wants_capex else "OPEX"
        subset = combined_df[combined_df["cost_type"] == cost_type]
        total = pd.to_numeric(subset[col], errors="coerce").sum() if col in subset.columns else None
        yr_label = year or plan_year
        return f"{cost_type} на {yr_label} год: {format_money(total, unit)} ({len(subset)} статей)."

    # branch comparison ("какая ветка/дочка тратит больше", "по филиалам")
    if any(w in q for w in ["ветк", "филиал", "по дочк", "какая дочк", "branch"]):
        col = year_col or plan_col
        g = combined_df.groupby("branch")[col].apply(lambda s: pd.to_numeric(s, errors="coerce").sum())
        g = g.sort_values(ascending=False)
        lines = [f"{b}: {format_money(v, unit)}" for b, v in g.items()]
        return "По подразделениям:\n" + "\n".join(lines)

    if any(w in q for w in ["без обоснован", "нет обоснован"]) and "обоснован" in q:
        if "missing_justification_flag" not in combined_df.columns:
            return "Обоснование не проверялось для этого файла."
        missing = combined_df[combined_df["missing_justification_flag"] == True]  # noqa: E712
        if missing.empty:
            return "Все статьи имеют заполненное обоснование."
        by_branch = missing.groupby("branch").size()
        lines = [f"{b}: {n} статей" for b, n in by_branch.items()]
        return f"Всего без обоснования: {len(missing)} статей.\n" + "\n".join(lines)

    if any(w in q for w in ["самые дорог", "самые крупн", "топ", "риск"]):
        col = year_col or plan_col
        top = combined_df.nlargest(5, col)[["item_name", "cost_type", "branch", col]]
        lines = [f"{r.item_name} ({r.cost_type}, {r.branch}): {format_money(getattr(r, col), unit)}" for r in top.itertuples()]
        return "Топ-5 самых крупных статей по всему файлу:\n" + "\n".join(lines)

    if year and year_col:
        total = pd.to_numeric(combined_df[year_col], errors="coerce").sum()
        label = "План" if year_col.startswith("plan_") else "Прогноз"
        return f"{label} на {year} год (OPEX+CAPEX, все подразделения): {format_money(total, unit)}."

    if any(w in q for w in ["сколько статей", "количество статей"]):
        return f"Всего заполненных статей по файлу: {len(combined_df)} (OPEX: {(combined_df['cost_type']=='OPEX').sum()}, CAPEX: {(combined_df['cost_type']=='CAPEX').sum()})."

    tokens = [t for t in re.split(r"[^а-яa-zё0-9]+", q) if len(t) > 3]
    if tokens:
        mask = pd.Series(False, index=combined_df.index)
        for col in ["item_name", "category", "subcategory", "counterparty"]:
            if col in combined_df.columns:
                for t in tokens:
                    mask |= combined_df[col].astype(str).str.lower().str.contains(t, na=False)
        matched = combined_df[mask]
        if not matched.empty:
            col = year_col or plan_col
            total = pd.to_numeric(matched[col], errors="coerce").sum() if col in matched.columns else None
            names = ", ".join(matched["item_name"].astype(str).head(5))
            return (f"Найдено {len(matched)} статей по всему файлу ({names}"
                    f"{'...' if len(matched) > 5 else ''}). Суммарно: {format_money(total, unit)}.")

    return None
