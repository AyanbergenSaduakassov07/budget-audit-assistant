"""
Text summary generation + a small rule-based query engine over one parsed
budget sheet. No LLM call is wired in here (no API key is available in this
environment) -- `answer_query()` is deliberately written as a single function
so it's a one-line swap to route unmatched questions to an LLM later
(see the docstring at the bottom).
"""

import re
import pandas as pd

from .parser import _best_baseline_col, year_trend, available_years


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

    n_anomalous_code = int(df["anomalous_code_flag"].sum()) if "anomalous_code_flag" in df else 0
    if n_anomalous_code:
        parts.append(
            f"{n_anomalous_code} статей имеют код, не соответствующий ожидаемому формату "
            f"(например IT-CA-01) -- это предупреждение о качестве данных, а не ошибка "
            f"расчёта; проверьте вручную, возможна опечатка или нестандартная нумерация."
        )

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

# Priority order for resolving "what happened in year X": prefer the
# committed plan if the form has one, otherwise the actual (факт), otherwise
# a forward-looking forecast -- matching how a human would read the question.
_YEAR_VALUE_PREFIXES = ("plan", "fact", "forecast")
_YEAR_VALUE_LABELS = {"plan": "План", "fact": "Факт", "forecast": "Прогноз"}


def _resolve_year_col(df, year):
    """First existing <prefix>_<year> column in priority order plan > fact >
    forecast, or None if the year isn't represented by any of the three."""
    if not year:
        return None
    for prefix in _YEAR_VALUE_PREFIXES:
        col = f"{prefix}_{year}"
        if col in df.columns:
            return col
    return None


# Phrases asking for a year-by-year breakdown instead of one year's figure,
# plus an explicit "с <year> до/по <year>" range. Used by chat_answer()'s
# CAPEX/OPEX branches to switch from a single year_col to a per-year table.
_MULTI_YEAR_MARKERS = ("по всем год", "за все год", "по годам", "по каждому году")
_YEAR_RANGE_RE = re.compile(r"с\s+(20\d{2})\s+(?:до|по)\s+(20\d{2})")


def _detect_multi_year_request(q):
    """(start, end) for an explicit 'с 2025 до 2027' range, True for a
    general 'all years' phrase with no explicit range, or False if the
    question isn't asking for a multi-year breakdown at all."""
    m = _YEAR_RANGE_RE.search(q)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return any(marker in q for marker in _MULTI_YEAR_MARKERS)


def _yoy_delta_str(current, previous):
    """' (Δ +N.N%)' vs. the previous year in the table, or '' for the first
    year in the range (nothing to compare against) or a zero/missing base."""
    if previous is None or not previous:
        return ""
    pct = (current - previous) / previous * 100
    return f" (Δ {pct:+.1f}%)"


def _multi_year_capex_opex_answer(combined_df, unit, multi_year):
    """Per-year OPEX/CAPEX/ratio/Δ% breakdown -- one line per year instead of
    a single year_col's totals. multi_year is either an explicit (start, end)
    tuple or True (every year present, from available_years())."""
    years_present = available_years(combined_df)
    if isinstance(multi_year, tuple):
        start, end = multi_year
        years = [y for y in years_present if start <= y <= end]
    else:
        years = list(years_present.keys())
    if not years:
        return "Нет годовых данных для запрошенного диапазона лет."

    lines = ["Разбивка OPEX/CAPEX по годам:"]
    prev_opex = prev_capex = None
    for year in years:
        col = _resolve_year_col(combined_df, year)
        if not col:
            continue
        opex = pd.to_numeric(combined_df[combined_df["cost_type"] == "OPEX"][col], errors="coerce").sum()
        capex = pd.to_numeric(combined_df[combined_df["cost_type"] == "CAPEX"][col], errors="coerce").sum()
        label = _YEAR_VALUE_LABELS[col.split("_", 1)[0]]
        ratio = f"{opex / capex:.1f}:1" if capex else "н/д"
        lines.append(
            f"{year} ({label}): OPEX = {format_money(opex, unit)}{_yoy_delta_str(opex, prev_opex)}, "
            f"CAPEX = {format_money(capex, unit)}{_yoy_delta_str(capex, prev_capex)}, "
            f"OPEX/CAPEX ~ {ratio}"
        )
        prev_opex, prev_capex = opex, capex

    if len(lines) == 1:
        return "Нет данных для расчёта разбивки по годам."
    return "\n".join(lines)


# --- shared keyword-matching helpers used by both answer_query() and chat_answer() --

# Bug C: "без обоснован"/"нет обоснован" was too narrow -- an exact adjacent
# phrase -- to catch real phrasings like "не имеют обоснования" or
# "обоснование отсутствует". Broadened to: the question mentions the topic
# ('обоснован') AND any negation word appears anywhere in the question.
_JUSTIFICATION_NEGATIONS = ("без", "нет", "не имеют", "отсутств", "не заполнен")


def _mentions_missing_justification(q):
    return "обоснован" in q and any(neg in q for neg in _JUSTIFICATION_NEGATIONS)


# Bug E: the "топ"/"самых"/"риск" ranking branches hardcoded nlargest(5).
# Parse an optional requested count adjacent to "топ"/"самых" ("топ-3",
# "топ 10", "3 самых"), capped at _MAX_TOP_N regardless of what's requested.
_TOP_N_RE = re.compile(r"топ[\s-]*(\d+)|(\d+)\s*(?:самых|топ)")
_MAX_TOP_N = 20


def _requested_top_n(q, default=5):
    m = _TOP_N_RE.search(q)
    if not m:
        return default
    n = int(m.group(1) or m.group(2))
    return max(1, min(n, _MAX_TOP_N))


# Bug #4 (routing): chat_answer()'s branches match on bare substrings with no
# check that the matched branch actually covers the WHOLE question. A
# question combining trigger words from more than one branch (a named-branch
# capex/opex comparison, two distinct years used as separate single-year
# asks rather than a range, a year + missing-justification clause, or an
# explicit conjunction joining two different keyword groups) gets a
# confident, wrong, PARTIAL answer from whichever branch's `if` fires
# first -- and the caller's smarter file_sql fallback (the only tier that can
# run a real multi-condition query) never runs, because the rule engine only
# returns None on a total miss, not a partial one. _looks_compound() bails
# out early instead, so chat_answer() returns None and app.py's existing
# fallback chain (chat_answer -> answer_over_file -> FALLBACK_REPLY) routes
# the question to the SQL/decomposition engine.
_RANKING_TOKENS = ("самые дорог", "самые крупн", "топ", "риск")
_COMPARISON_TOKENS = ("сравни", "сравнение")
_CAPEX_OPEX_TOKENS = ("capex", "капекс", "капвлож", "opex", "опекс", "операционн")
_BRANCH_GROUP_TOKENS = ("ветк", "филиал", "по дочк", "какая дочк", "branch")
_COMPOUND_CONJUNCTIONS = (" и ", "а также", "при этом")


def _mentioned_branch_names(q, combined_df):
    """Distinct actual branch values from THIS file's data that literally
    appear in the question -- e.g. 'го' and 'аф' in 'сравни capex го и аф'."""
    if combined_df is None or "branch" not in combined_df.columns:
        return set()
    found = set()
    for name in combined_df["branch"].dropna().astype(str).unique():
        name_l = name.lower().strip()
        if name_l and re.search(rf"(?<![а-яa-zё0-9]){re.escape(name_l)}(?![а-яa-zё0-9])", q):
            found.add(name_l)
    return found


def _looks_compound(q, combined_df):
    """True if `q` combines signals from more than one chat_answer() branch,
    so the caller should bail out to None (routing to the SQL/decomposition
    fallback) instead of letting whichever branch matches first answer only
    part of the question."""
    years = set(re.findall(r"20\d{2}", q))
    # An explicit "с 2026 до 2027" range or a general "по всем годам"/"за все
    # года" marker legitimately mentions 2+ years -- that's the multi-year
    # breakdown branch's job, not a compound question.
    multi_year = _detect_multi_year_request(q)
    if len(years) > 1 and not multi_year:
        return True

    has_justification = "обоснован" in q
    has_ranking = any(t in q for t in _RANKING_TOKENS)
    has_comparison = any(t in q for t in _COMPARISON_TOKENS)
    has_capex_opex = any(t in q for t in _CAPEX_OPEX_TOKENS)
    has_branch_group = any(t in q for t in _BRANCH_GROUP_TOKENS)
    branch_names = _mentioned_branch_names(q, combined_df)

    if years and has_justification:
        return True
    if has_comparison and has_ranking:
        return True
    # "сравни CAPEX ГО и АФ": naming 2+ specific branches alongside a
    # cost-type keyword means "compare these branches", not the whole-
    # company OPEX/CAPEX split the plain comparison branch answers.
    if has_capex_opex and len(branch_names) >= 2:
        return True
    if has_comparison and has_capex_opex and has_branch_group:
        return True

    matched_groups = sum([has_capex_opex, has_branch_group, has_justification, has_ranking])
    if matched_groups >= 2 and any(c in q for c in _COMPOUND_CONJUNCTIONS):
        return True

    return False


# A short "reformat what you just showed me" follow-up ("в виде таблицы",
# "покажи списком") carries no business keyword of its own. chat_answer() is
# stateless -- it has no idea what "it" refers to -- so letting one of these
# reach the generic keyword-search fallback below risks a coincidental
# substring match (observed live: "в виде таблицы" tokenizes to "виде" +
# "таблицы", and "виде" matched unrelated item names containing "видения",
# e.g. "Услуги кабельного телевидения"). Bail out to None here so the caller
# (app.answer_question) routes straight to the history-aware LLM fallback,
# which at least has the prior answer to reformat.
_REFORMAT_ONLY_TRIGGERS = (
    "в виде таблиц", "таблицей", "в таблице", "списком", "по-другому",
    "по другому", "переформат", "в виде списка", "построчно",
)


def _is_reformat_only_request(q: str) -> bool:
    return any(t in q for t in _REFORMAT_ONLY_TRIGGERS)


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
    year_col = _resolve_year_col(df, year)

    if _mentions_missing_justification(q):
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
        n = _requested_top_n(q)
        top = df.nlargest(n, plan_col)[["item_name", plan_col]]
        lines = [f"{r.item_name}: {format_money(getattr(r, plan_col), unit)}" for r in top.itertuples()]
        return f"Топ-{n} самых крупных статей:\n" + "\n".join(lines)

    if any(w in q for w in ["сколько статей", "количество статей", "число статей"]):
        return f"Заполненных статей расхода: {len(df)}."

    # keyword search across item_name / category / subcategory / counterparty
    # -- tried BEFORE the plain year-total branch below (Bug B): a question
    # naming both a year AND some other filter (counterparty/category/item
    # keyword) must be answered by that filter, not silently fall back to the
    # whole year's total just because a year was also mentioned.
    tokens = [t for t in re.split(r"[^а-яa-zё0-9]+", q) if len(t) > 3]
    if tokens:
        mask = pd.Series(False, index=df.index)
        for col in ["item_name", "category", "subcategory", "counterparty"]:
            if col in df.columns:
                for t in tokens:
                    mask |= df[col].astype(str).str.lower().str.contains(t, na=False)
        matched = df[mask]
        if not matched.empty:
            col = year_col or f"plan_{stats['plan_year']}"
            total = pd.to_numeric(matched[col], errors="coerce").sum() if col in matched.columns else None
            names = ", ".join(matched["item_name"].astype(str).head(5))
            return (
                f"Найдено {len(matched)} совпадающих статей ({names}"
                f"{'...' if len(matched) > 5 else ''}). "
                f"Суммарный план: {format_money(total, unit)}."
            )

    # Only use the plain year-total branch when the token search above found
    # nothing to filter on.
    if year_col:
        total = pd.to_numeric(df[year_col], errors="coerce").sum()
        label = _YEAR_VALUE_LABELS[year_col.split("_", 1)[0]]
        return f"{label} на {year} год: {format_money(total, unit)}."

    return None  # no rule matched -- plug in an LLM fallback here


# --- Where to add a real LLM later --------------------------------------
# def answer_query_llm(question, df, meta, api_key):
#     context = df.to_csv(index=False)
#     prompt = f"Данные бюджета:\n{context}\n\nВопрос: {question}\nОтветь кратко на русском."
#     # call the LLM provider here, return response.text


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
    if _looks_compound(q, combined_df):
        return None
    if _is_reformat_only_request(q):
        return None

    unit = meta.get("unit", "")
    plan_year = combined_df["plan_year"].dropna().iloc[0] if combined_df["plan_year"].notna().any() else None
    plan_col = f"plan_{plan_year}" if plan_year else None

    year_match = re.search(r"(20\d{2})", q)
    year = int(year_match.group(1)) if year_match else None
    year_col = _resolve_year_col(combined_df, year)

    wants_capex = "capex" in q or "капекс" in q or "капвлож" in q
    wants_opex = "opex" in q or "опекс" in q or "операционн" in q
    wants_comparison = "сравни" in q or "сравнение" in q

    # Multi-year OPEX/CAPEX breakdown ("...по всем годам" / "...за все года" /
    # "...по годам" / an explicit "с 2025 до 2027" range) instead of the
    # single-year_col branches below. "сравни" without an explicit capex/opex
    # keyword still routes here -- comparing OPEX vs CAPEX is the only thing
    # "сравни" means anywhere else in this function.
    multi_year = _detect_multi_year_request(q)
    if multi_year and (wants_capex or wants_opex or wants_comparison):
        return _multi_year_capex_opex_answer(combined_df, unit, multi_year)

    # CAPEX vs OPEX comparison
    if wants_comparison and ("capex" in q or "opex" in q or "капекс" in q or "опекс" in q):
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

    if _mentions_missing_justification(q):
        if "missing_justification_flag" not in combined_df.columns:
            return "Обоснование не проверялось для этого файла."
        missing = combined_df[combined_df["missing_justification_flag"] == True]  # noqa: E712
        if missing.empty:
            return "Все статьи имеют заполненное обоснование."
        by_branch = missing.groupby("branch").size()
        lines = [f"{b}: {n} статей" for b, n in by_branch.items()]
        return f"Всего без обоснования: {len(missing)} статей.\n" + "\n".join(lines)

    if any(w in q for w in ["самые дорог", "самые крупн", "топ", "риск"]):
        n = _requested_top_n(q)
        col = year_col or plan_col
        top = combined_df.nlargest(n, col)[["item_name", "cost_type", "branch", col]]
        lines = [f"{r.item_name} ({r.cost_type}, {r.branch}): {format_money(getattr(r, col), unit)}" for r in top.itertuples()]
        return f"Топ-{n} самых крупных статей по всему файлу:\n" + "\n".join(lines)

    # keyword search across item_name / category / subcategory / counterparty
    # -- tried BEFORE the plain year-total branch below (Bug B): a question
    # naming both a year AND some other filter (counterparty/category/item
    # keyword) must be answered by that filter, not silently fall back to the
    # whole-file year total just because a year was also mentioned.
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

    # Only use the plain year-total branch when the token search above found
    # nothing to filter on.
    if year and year_col:
        total = pd.to_numeric(combined_df[year_col], errors="coerce").sum()
        label = _YEAR_VALUE_LABELS[year_col.split("_", 1)[0]]
        return f"{label} на {year} год (OPEX+CAPEX, все подразделения): {format_money(total, unit)}."

    if any(w in q for w in ["сколько статей", "количество статей"]):
        return f"Всего заполненных статей по файлу: {len(combined_df)} (OPEX: {(combined_df['cost_type']=='OPEX').sum()}, CAPEX: {(combined_df['cost_type']=='CAPEX').sum()})."

    # A specific year was named, but this form carries no plan_/forecast_ column
    # for it (common on subsidiary forms with a different campaign year). Answer
    # precisely with the years that ARE present instead of the generic fallback.
    if year is not None and year_col is None:
        avail = available_years(combined_df)
        years_list = ", ".join(str(y) for y in avail) if avail else "нет годовых колонок"
        return (f"Данных за {year} год в этом файле нет. "
                f"Доступные годы: {years_list}.")

    # General "what years / prior years are in here" question with no specific
    # year mentioned — answer from year_trend() (same source as the
    # «Динамика по годам» chart), listing every факт/план/прогноз year present.
    # A comparison request ("сравни их") is deliberately NOT caught here: it
    # needs real aggregation, so it falls through to the SQL fallback in app.py.
    has_year_word = "год" in q or "лет" in q
    wants_comparison = "сравн" in q
    if year is None and not wants_comparison and (
        ("прошл" in q and has_year_word)
        or ("предыдущ" in q and has_year_word)
        or "истори" in q
        or "какие года" in q or "какие годы" in q
        or "доступные год" in q
    ):
        yt = year_trend(combined_df)
        if yt.empty:
            return "В этом файле нет годовых колонок (план/факт/прогноз) для сравнения по годам."
        lines = [f"{r.Год} — {r.Тип}: {format_money(r.Сумма, unit)}" for r in yt.itertuples()]
        return "Годы, представленные в файле:\n" + "\n".join(lines)

    return None
