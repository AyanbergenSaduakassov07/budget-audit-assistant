"""ACM IT Budget Auditor — single-page Streamlit app.

This is the ONLY file allowed to import streamlit (SPEC.md architecture rule).
All parsing/analysis math lives in core.budget_auditor.{parser,analysis};
app.py only reshapes, renames, formats, and renders what that layer returns.

Page flow (fixed order):
  1. One .xlsx uploader (форма ИТ-расходов ИТ-бюджет).
  2. Auto-generated text description of the whole workbook.
  3. CAPEX vs OPEX comparison table + bar chart for the campaign year.
  4. Three flagged-статьи views: резкое изменение г/г (порог — слайдер),
     без обоснования, топ-10% по стоимости.
  5. Chat over the ENTIRE file (all sheets, branches, OPEX+CAPEX).
"""

import logging
import os
import tempfile

import pandas as pd
import streamlit as st

from core.budget_auditor.parser import (
    parse_workbook,
    flag_rows,
    aggregate_workbook,
    capex_opex_table,
    year_trend,
    _best_baseline_col,
)
from core.budget_auditor.analysis import workbook_summary_text, year_trend_text, chat_answer

logger = logging.getLogger(__name__)

# --- palette (SPEC design system) --------------------------------------------
NAVY = "#1F3A5F"
BG = "#F6F4EF"
# Fixed per-series colors for the year-trend chart (color follows the entity,
# never the series count): план=navy, факт=green, прогноз=amber.
TREND_COLORS = {"план": "#1F3A5F", "факт": "#2E7D32", "прогноз": "#F59E0B"}

DEFAULT_THRESHOLD = 15

# Display labels for the flagged-rows tables (raw column -> Russian header).
FLAG_VIEW_LABELS = {
    "branch": "Подразделение",
    "cost_type": "Тип",
    "statya_code": "Статья",
    "statya_name": "Наименование статьи",
    "item_name": "Позиция",
    "counterparty": "Контрагент",
    "delta_abs": "Δ абс.",
    "delta_pct": "Δ %",
    "justification": "Обоснование",
}

MONEY_FMT = "{:,.0f}"


# --- rendering helpers (presentation only, no math) ---------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp { background-color: #F6F4EF; }
        h1, h2, h3, h4, h5, h6 { color: #1F3A5F !important; }
        p, label, span, div { color: #2C2C2A; }
        .stMetric label { color: #5F5E5A !important; font-size: 13px; }
        .stMetric [data-testid="stMetricValue"] { color: #1F3A5F !important; font-size: 22px; font-weight: 700; }
        .stCaption { color: #5F5E5A !important; }
        section[data-testid="stSidebar"] { background-color: #FFFFFF; }
        [data-testid="stDataFrame"] { background-color: #FFFFFF; border-radius: 6px; }
        .app-header { background: #1F3A5F; padding: 24px 28px; border-radius: 10px; margin-bottom: 20px; }
        .app-header-title { color: #FFFFFF !important; margin: 0; font-size: 22px; font-weight: 700; }
        .app-header-sub { color: #D5D2CC !important; margin: 6px 0 0; font-size: 14px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _money(value) -> str:
    """Space-grouped thousands (Russian convention) — 'н/д' for missing values.

    The NA guard must run before any float() call: pd.NA is not an int/float
    instance, and Styler.format() feeds these raw cell values in lazily at
    render time — any file without a prior-year baseline puts pd.NA in the
    delta columns.
    """
    if value is None or pd.isna(value):
        return "н/д"
    return MONEY_FMT.format(float(value)).replace(",", " ")


def _pct(value) -> str:
    if value is None or pd.isna(value):
        return "н/д"
    return f"{value:+.1f}%"


def _share(value) -> str:
    if value is None or pd.isna(value):
        return "н/д"
    return f"{value:.1f}%"


@st.cache_data(show_spinner=False)
def _cached_parse(file_bytes: bytes):
    """parse_workbook() over the uploaded bytes, cached on those bytes."""
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        return parse_workbook(tmp_path)
    finally:
        os.unlink(tmp_path)


def build_flag_view(subset: pd.DataFrame, plan_col: str | None) -> tuple[pd.DataFrame, dict]:
    """Pure part of a flagged-статьи table: select display columns, Russian
    headers, and the Styler format map. Split from render_flag_view so tests
    can render the exact production Styler without a Streamlit runtime."""
    labels = dict(FLAG_VIEW_LABELS)
    if plan_col:
        labels[plan_col] = f"План {plan_col.removeprefix('plan_')}"
    show = [c for c in
            ["branch", "cost_type", "statya_code", "item_name", "counterparty",
             plan_col, "delta_abs", "delta_pct", "justification"]
            if c and c in subset.columns]
    disp = subset[show].rename(columns=labels).reset_index(drop=True)
    fmt = {}
    if plan_col and labels[plan_col] in disp.columns:
        fmt[labels[plan_col]] = _money
    if "Δ абс." in disp.columns:
        fmt["Δ абс."] = _money
    if "Δ %" in disp.columns:
        fmt["Δ %"] = _pct
    return disp, fmt


def render_flag_view(subset: pd.DataFrame, plan_col: str | None, empty_msg: str) -> None:
    if subset.empty:
        st.success(empty_msg)
        return
    disp, fmt = build_flag_view(subset, plan_col)
    st.dataframe(disp.style.format(fmt), use_container_width=True, hide_index=True)


# --- page ----------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Аудит IT-бюджетов компании", page_icon=None)
inject_css()

st.markdown(
    '<div class="app-header">'
    '<div class="app-header-title">Аудит IT-бюджетов компании</div>'
    '<p class="app-header-sub">Загрузите один файл формы ИТ-расходов — система '
    'разберёт его целиком, опишет текстом, сравнит CAPEX/OPEX, найдёт проблемные '
    'статьи и ответит на вопросы по всему файлу</p></div>',
    unsafe_allow_html=True,
)

# ---------- 1. Single file uploader ----------
uploaded = st.file_uploader("Excel-файл формы ИТ-расходов (.xlsx)", type=["xlsx"])
if uploaded is None:
    st.info("Загрузите файл, чтобы начать анализ.")
    st.stop()

try:
    meta, sheets = _cached_parse(uploaded.getvalue())
except Exception as e:
    logger.exception("workbook parse failed")
    st.error(f"Не удалось разобрать файл: {e}")
    st.stop()

usable_sheets = {sn: d for sn, d in sheets.items() if d["df"] is not None and not d["df"].empty}
empty_sheets = [sn for sn in sheets if sn not in usable_sheets]

if not usable_sheets:
    st.warning(
        "Ни в одном листе не найдено заполненных статей расхода. "
        "Проверьте, что это действительно форма ИТ-расходов компании."
    )
    st.stop()

# The threshold slider is rendered in section 4, but the flag columns it controls
# feed the description in section 2. Streamlit reruns top-to-bottom with widget
# state already updated in session_state, so reading it here is consistent on
# every rerun; the first run simply uses the slider's default.
threshold = float(st.session_state.get("yoy_threshold", DEFAULT_THRESHOLD))

flagged_sheets = {
    sn: {"df": flag_rows(d["df"], d["stats"], threshold_pct=threshold), "stats": d["stats"]}
    for sn, d in usable_sheets.items()
}
combined = aggregate_workbook(flagged_sheets)

plan_year = (
    combined["plan_year"].dropna().iloc[0]
    if not combined.empty and combined["plan_year"].notna().any()
    else None
)
plan_col = f"plan_{plan_year}" if plan_year else None

# ---------- 2. Auto-generated description ----------
st.header("1. Описание файла")
st.write(workbook_summary_text(meta, flagged_sheets))
if empty_sheets:
    st.warning(
        "Пустые/незаполненные листы (только шаблон, без данных): "
        + ", ".join(empty_sheets)
    )

# ---------- 3. CAPEX vs OPEX ----------
st.header(f"2. CAPEX vs OPEX — {plan_year if plan_year else 'год кампании'}")
cvo = capex_opex_table(combined)
if cvo.empty:
    st.info("Недостаточно данных для сравнения CAPEX/OPEX.")
else:
    plan_label = next((c for c in cvo.columns if c.startswith("План ")), None)
    fmt = {}
    if plan_label:
        fmt[plan_label] = _money
    if "Доля" in cvo.columns:
        fmt["Доля"] = _share
    if "База пред. года" in cvo.columns:
        fmt["База пред. года"] = _money
    if "Δ %" in cvo.columns:
        fmt["Δ %"] = _pct
    col_table, col_chart = st.columns([1, 2])
    with col_table:
        st.dataframe(cvo.style.format(fmt), use_container_width=True, hide_index=True)
        if "Δ %" not in cvo.columns:
            st.caption(
                "Δ год-к-году не показана: в форме нет колонок прошлого года "
                "(структурный пробел формы, не ошибка загрузки)."
            )
    with col_chart:
        if plan_label:
            st.bar_chart(cvo.set_index("Тип")[plan_label], color=NAVY)

# ---------- 3b. Динамика по годам ----------
st.header("2b. Динамика по годам")
trend = year_trend(combined)
if trend.empty:
    st.info("Недостаточно годовых колонок для построения тренда.")
else:
    # str years so the axis doesn't render 2026 as "2,026"
    wide = (
        trend.assign(Год=trend["Год"].astype(str))
        .pivot_table(index="Год", columns="Тип", values="Сумма", aggfunc="sum")
    )
    st.line_chart(wide, color=[TREND_COLORS.get(c, NAVY) for c in wide.columns])
    trend_note = year_trend_text(combined)
    if trend["unit_mismatch_suspected"].any():
        st.warning(trend_note)
    else:
        st.caption(trend_note)

# ---------- 4. Flagged статьи (workbook-wide) ----------
st.header("3. Проблемные и рискованные статьи (по всему файлу)")
st.slider(
    "Порог резкого изменения год-к-году, %",
    min_value=5, max_value=50, value=DEFAULT_THRESHOLD, step=1,
    key="yoy_threshold",
)

no_prior_sheets = [
    sn for sn, d in flagged_sheets.items()
    if _best_baseline_col(d["df"], d["stats"].get("plan_year")) is None
]
no_just_sheets = [
    sn for sn, d in flagged_sheets.items() if not d["stats"]["has_justification"]
]

tab_change, tab_just, tab_price = st.tabs(
    ["Резкое изменение г/г", "Без обоснования", "Топ-10% по стоимости"]
)

with tab_change:
    d = combined[combined["high_change_flag"]] if "high_change_flag" in combined else pd.DataFrame()
    st.write(f"{len(d)} статей с изменением более {threshold:.0f}% к прошлому году")
    if no_prior_sheets:
        st.caption(
            "Листы без данных прошлого года (сравнение г/г структурно невозможно): "
            + ", ".join(no_prior_sheets)
        )
    render_flag_view(d, plan_col, "Резких изменений год-к-году не найдено.")

with tab_just:
    d = combined[combined["missing_justification_flag"]] if "missing_justification_flag" in combined else pd.DataFrame()
    st.write(f"{len(d)} статей без обоснования по всему файлу")
    if no_just_sheets:
        st.warning(
            "В этих листах колонка «Обоснование» отсутствует целиком — все их статьи "
            "считаются необоснованными: " + ", ".join(no_just_sheets)
        )
    render_flag_view(d, plan_col, "Все статьи имеют заполненное обоснование.")

with tab_price:
    d = combined[combined["high_price_flag"]] if "high_price_flag" in combined else pd.DataFrame()
    st.write(f"{len(d)} статей в верхних 10% по стоимости (внутри своего листа)")
    render_flag_view(d, plan_col, "Ценовых выбросов не найдено.")

# ---------- 5. Chat over the whole file ----------
st.divider()
st.header("4. Чат по этому файлу")
st.caption(
    "Вопросы по всему файлу сразу — все листы, подразделения (ГО/АФ/КазНИПИ), "
    "OPEX и CAPEX: «сколько CAPEX в 2027», «сравни OPEX и CAPEX», "
    "«какая ветка тратит больше», «топ самых дорогих статей»"
)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if st.session_state["chat_history"] and st.button("Очистить чат"):
    st.session_state["chat_history"] = []
    st.rerun()

for entry in st.session_state["chat_history"]:
    with st.chat_message(entry["role"]):
        st.write(entry["content"])

question = st.chat_input("Введите вопрос по файлу")
if question:
    st.session_state["chat_history"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    reply = chat_answer(question, combined, meta)
    if reply is None:
        reply = (
            "Не нашёл точного ответа на этот вопрос. Попробуйте указать год "
            "(например, «план на 2026») или ключевое слово статьи расхода."
        )
    st.session_state["chat_history"].append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.write(reply)

st.divider()
st.caption("Аудит IT-бюджетов · MVP v3.0")
