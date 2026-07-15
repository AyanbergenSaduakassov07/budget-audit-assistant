"""Streamlit UI for the procurement comparison tool.

This is the ONLY file allowed to import streamlit (SPEC.md architecture rule).
It contains no arithmetic or comparison logic — every number shown here comes
from core.compare / core.schema. app.py only reshapes, renames, formats, and
renders what core returns.
"""

import math
import os
import tempfile
from io import BytesIO

import pandas as pd
import streamlit as st

from core.synth import generate, generate_group
from core.schema import normalize
from core.compare import compare
from core.warehouse import (
    build_warehouse,
    get_group_summary,
    get_yearly_totals,
    get_subsidiary_totals_for_year,
    get_subsidiary_totals,
)
from core.query_engine import answer
from core.corporate_ingest import extract_workbook
from core.budget_audit import run_full_audit
from core.budget_narrator import narrate_audit
from core.conclusion_generator import generate_conclusion

# Most recent year covered by the synthetic group data (see synth._GROUP_YEARS).
LATEST_YEAR = 2026

# --- palette (SPEC design system) -------------------------------------------
NAVY = "#1F3A5F"
BG = "#F6F4EF"
ANOMALY = "#FFCDD2"
ONLY_A = "#FFF3C4"
ONLY_B = "#FFE0B2"
POSITIVE = "#C8E6C9"
NEGATIVE = "#FFCDD2"

# --- audit-tab palette (SPEC design system, audit tab) ------------------------
AUDIT_SECTION_BG = {"АТК": "#FFCDD2", "уточнение": "#FFF3C4", "согласовано": "#C8E6C9"}
AUDIT_SECTION_TEXT = {"АТК": "#7F1D1D", "уточнение": "#78350F", "согласовано": "#1B5E20"}
AUDIT_SECTION_ICON = {"АТК": "🔴", "уточнение": "🟡", "согласовано": "🟢"}
AUDIT_SECTION_LABEL = {
    "АТК": "На рассмотрение АТК",
    "уточнение": "Требуют уточнения",
    "согласовано": "Согласовано без замечаний",
}
# class= counterparts of AUDIT_SECTION_TEXT for st.markdown HTML (see inject_audit_css
# docstring note: inline style="color:..." gets sanitized away, class="..." doesn't).
AUDIT_SECTION_TEXT_CLASS = {
    "АТК": "audit-c-atk-text",
    "уточнение": "audit-c-clarify-text",
    "согласовано": "audit-c-approved-text",
}
RISK_TEXT_CLASS = {
    "высокий": "audit-c-risk-high",
    "средний": "audit-c-risk-medium",
    "низкий": "audit-c-risk-low",
}
RISK_HIGH = "#C62828"
RISK_MEDIUM = "#F59E0B"
RISK_LOW = "#2E7D32"
RISK_UNKNOWN = "#5F5E5A"

# --- Russian display labels --------------------------------------------------
SHEET_LABELS = {
    "item_name": "Наименование",
    "supplier": "Поставщик",
    "category": "Категория",
    "qty": "Кол-во",
    "unit_price": "Цена за ед., ₸",
    "subtotal": "Сумма, ₸",
}
DELTA_LABELS = {
    "item_name": "Наименование",
    "supplier": "Поставщик",
    "unit_price_a": "Цена А, ₸",
    "unit_price_b": "Цена Б, ₸",
    "pct_delta_price": "Δ цены, %",
    "delta_subtotal": "Δ сумма, ₸",
    "contribution_pct": "Вклад, %",
    "cumulative_pct": "Накопл., %",
    "severity_score": "Северити",
    "is_price_anomaly": "Аномалия",
}
ROLLUP_LABELS = {
    "subtotal_a": "Сумма А, ₸",
    "subtotal_b": "Сумма Б, ₸",
    "delta_subtotal": "Δ, ₸",
    "pct_delta": "Δ, %",
}

MONEY_FMT = "{:,.0f}"
PCT_FMT = "{:+.2f}"

# --- column-mapping config for uploaded files --------------------------------
# Case-insensitive, whitespace-stripped aliases used to auto-guess which
# uploaded column corresponds to each required schema field.
COLUMN_GUESS = {
    "item_name": ["item_name", "наименование", "товар", "название", "name", "item", "product", "позиция"],
    "supplier": ["supplier", "поставщик", "vendor", "поставщики", "провайдер"],
    "category": ["category", "категория", "тип", "type", "группа"],
    "qty": ["qty", "quantity", "кол-во", "количество", "count", "объем", "объём"],
    "unit_price": ["unit_price", "цена", "price", "цена за ед", "цена за единицу", "стоимость единицы", "unit cost"],
    "subtotal": ["subtotal", "сумма", "итого", "total", "amount", "сумма итого", "стоимость"],
}
MAPPING_FIELD_LABELS = {
    "item_name": "Наименование товара",
    "supplier": "Поставщик",
    "category": "Категория",
    "qty": "Количество",
    "unit_price": "Цена за единицу (₸)",
    "subtotal": "Сумма (₸)",
}
NOT_SELECTED = "— не выбрано —"


# --- rendering helpers (presentation only, no math) --------------------------
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
        .stInfo { background-color: #E8F4FD !important; color: #1F3A5F !important; }
        section[data-testid="stSidebar"] { background-color: #FFFFFF; }
        [data-testid="stDataFrame"] { background-color: #FFFFFF; border-radius: 6px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_audit_css() -> None:
    """Design-system CSS scoped to the Аудит бюджета tab (forces light theme)."""
    st.markdown(
        """
        <style>
        /* force light background regardless of system dark mode */
        .stApp { background-color: #F6F4EF !important; color: #2C2C2A !important; }
        h1, h2, h3, h4 { color: #1F3A5F !important; font-weight: 700 !important; }
        p, label, span, div { color: #2C2C2A !important; }
        .stMetric label { color: #5F5E5A !important; font-size: 13px !important; }
        .stMetric [data-testid="stMetricValue"] { color: #1F3A5F !important; font-size: 24px !important; font-weight: 700 !important; }
        /* expander headers */
        .streamlit-expanderHeader { background-color: #1F3A5F !important; color: white !important; border-radius: 6px !important; font-weight: 600 !important; }
        /* sort dropdown */
        div[data-baseweb="popover"] { background-color: #FFFFFF !important; }
        div[data-baseweb="popover"] * { color: #2C2C2A !important; background-color: #FFFFFF !important; }
        /* dataframe internals */
        [data-testid="stDataFrame"] { background-color: #FFFFFF !important; border-radius: 6px !important; border: 1px solid #D5D2CC !important; }
        :root { color-scheme: light !important; }
        [data-testid="stAppViewContainer"] { background-color: #F6F4EF !important; }
        /* info/warning/error boxes */
        .stInfo { background-color: #EFF6FF !important; color: #1F3A5F !important; border-left: 4px solid #1F3A5F !important; }
        .stWarning { background-color: #FFF3C4 !important; color: #92400E !important; border-left: 4px solid #F59E0B !important; }
        .stError { background-color: #FFCDD2 !important; color: #7F1D1D !important; border-left: 4px solid #C62828 !important; }
        .stSuccess { background-color: #C8E6C9 !important; color: #1B5E20 !important; border-left: 4px solid #2E7D32 !important; }
        /* Streamlit's markdown sanitizer strips the `color` CSS property from any
           inline style="..." attribute (background/margin/font-size survive fine,
           color specifically does not, even with !important). These utility
           classes are the workaround: apply color via class="..." instead of
           style="color:...", since class selectors aren't touched by that
           sanitization pass and outrank the bare-tag rules above by specificity. */
        .audit-c-white { color: #FFFFFF !important; }
        .audit-c-subtitle { color: #D5D2CC !important; }
        .audit-c-navy { color: #1F3A5F !important; }
        .audit-c-body { color: #2C2C2A !important; }
        .audit-c-atk-text { color: #7F1D1D !important; }
        .audit-c-clarify-text { color: #78350F !important; }
        .audit-c-approved-text { color: #1B5E20 !important; }
        .audit-c-risk-high { color: #C62828 !important; }
        .audit-c-risk-medium { color: #F59E0B !important; }
        .audit-c-risk-low { color: #2E7D32 !important; }
        .audit-c-risk-unknown { color: #5F5E5A !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _audit_money(value) -> str:
    """Space-grouped thousands (Russian convention) — '—' for missing values."""
    if value is None or (isinstance(value, (int, float)) and pd.isna(value)):
        return "—"
    return f"{float(value):,.0f}".replace(",", " ")


def _audit_pct(value) -> str:
    """Signed percentage; handles the delta_pct == inf (новая статья) case."""
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isinf(value):
            return "новая статья"
        if pd.isna(value):
            return "—"
    return f"{value:+.1f}%"


def _safe_col(df: pd.DataFrame, name: str, empty=""):
    """df[name] if present, else a column of `empty` — keeps display code column-agnostic."""
    if name in df.columns:
        return df[name]
    return pd.Series([empty] * len(df), index=df.index)


def _safe_text_col(df: pd.DataFrame, name: str):
    """_safe_col, but also blanks out None/NaN cells within an existing column to '—'."""
    series = _safe_col(df, name)
    return series.where(series.notna() & (series.astype(str).str.strip() != ""), "—")


def _build_sheet(matched: pd.DataFrame, only_in: pd.DataFrame, side: str) -> pd.DataFrame:
    """Reassemble one full normalized sheet from core's compare() output.

    Values (including arithmetic_error) come straight from core — this only
    selects, renames, and concatenates for display.
    """
    m = matched[
        ["item_name", "supplier", "category",
         f"qty_{side}", f"unit_price_{side}", f"subtotal_{side}", f"arithmetic_error_{side}"]
    ].copy()
    m.columns = ["item_name", "supplier", "category", "qty", "unit_price", "subtotal", "arithmetic_error"]
    o = only_in[
        ["item_name", "supplier", "category", "qty", "unit_price", "subtotal", "arithmetic_error"]
    ].copy()
    return pd.concat([m, o], ignore_index=True)


def render_sheet(df: pd.DataFrame) -> None:
    """Split-screen sheet: 6 Russian columns, arithmetic-error rows in red."""
    mask = df["arithmetic_error"].reset_index(drop=True)
    disp = df[list(SHEET_LABELS)].rename(columns=SHEET_LABELS).reset_index(drop=True)

    def highlight(row):
        bg = ANOMALY if bool(mask.iloc[row.name]) else ""
        return [f"background-color: {bg}"] * len(row)

    styled = disp.style.apply(highlight, axis=1).format(
        {SHEET_LABELS["unit_price"]: MONEY_FMT,
         SHEET_LABELS["subtotal"]: MONEY_FMT,
         SHEET_LABELS["qty"]: MONEY_FMT}
    )
    st.dataframe(styled, use_container_width=True)


def render_delta_table(matched: pd.DataFrame) -> None:
    disp = matched[list(DELTA_LABELS)].rename(columns=DELTA_LABELS).reset_index(drop=True)
    disp[DELTA_LABELS["is_price_anomaly"]] = disp[DELTA_LABELS["is_price_anomaly"]].map(
        {True: "Да", False: "Нет"}
    )

    def highlight(row):
        i = row.name
        anomaly = bool(matched["is_price_anomaly"].iloc[i])
        pct = matched["pct_delta_price"].iloc[i]
        if anomaly:
            bg = ANOMALY
        elif pct > 0:
            bg = POSITIVE
        else:
            bg = ""  # negative and non-anomaly: no highlight
        return [f"background-color: {bg}"] * len(row)

    styled = disp.style.apply(highlight, axis=1).format(
        {
            DELTA_LABELS["unit_price_a"]: MONEY_FMT,
            DELTA_LABELS["unit_price_b"]: MONEY_FMT,
            DELTA_LABELS["delta_subtotal"]: MONEY_FMT,
            DELTA_LABELS["pct_delta_price"]: PCT_FMT,
            DELTA_LABELS["contribution_pct"]: PCT_FMT,
            DELTA_LABELS["cumulative_pct"]: PCT_FMT,
            DELTA_LABELS["severity_score"]: "{:.2f}",
        }
    )
    st.dataframe(styled, use_container_width=True)


def render_only_in(df: pd.DataFrame, bg_color: str) -> None:
    disp = df[list(SHEET_LABELS)].rename(columns=SHEET_LABELS).reset_index(drop=True)
    styled = disp.style.apply(
        lambda row: [f"background-color: {bg_color}"] * len(row), axis=1
    ).format(
        {SHEET_LABELS["unit_price"]: MONEY_FMT,
         SHEET_LABELS["subtotal"]: MONEY_FMT,
         SHEET_LABELS["qty"]: MONEY_FMT}
    )
    st.dataframe(styled, use_container_width=True)


def render_rollup(rollup: pd.DataFrame, key_col: str, key_label: str) -> None:
    ordered = rollup.sort_values(
        "delta_subtotal", key=lambda s: s.abs(), ascending=False
    ).reset_index(drop=True)
    labels = {key_col: key_label, **ROLLUP_LABELS}
    disp = ordered[list(labels)].rename(columns=labels)

    def color_pct(val):
        if val > 0:
            return f"background-color: {NEGATIVE}"  # spend up = red
        if val < 0:
            return f"background-color: {POSITIVE}"  # spend down = green
        return ""

    styled = disp.style.map(color_pct, subset=[ROLLUP_LABELS["pct_delta"]]).format(
        {
            ROLLUP_LABELS["subtotal_a"]: MONEY_FMT,
            ROLLUP_LABELS["subtotal_b"]: MONEY_FMT,
            ROLLUP_LABELS["delta_subtotal"]: MONEY_FMT,
            ROLLUP_LABELS["pct_delta"]: PCT_FMT,
        }
    )
    st.dataframe(styled, use_container_width=True)


def map_columns(df: pd.DataFrame, file_label: str) -> pd.DataFrame:
    """Map an uploaded file's columns onto the required schema fields.

    Presentation-layer only: auto-guesses a mapping, lets the user confirm it via
    selectboxes, then renames/subsets columns so the result can be handed to
    core.schema.normalize(). No arithmetic here.
    """
    actual_cols = list(df.columns)
    # normalized (lower, stripped) actual name -> real column name, for guessing
    lookup = {str(c).strip().lower(): c for c in actual_cols}

    with st.expander("Сопоставление колонок — " + file_label, expanded=True):
        st.info(
            "Укажите, какие колонки вашего файла соответствуют нужным полям. "
            "Система попыталась определить их автоматически."
        )
        selection = {}
        for field, aliases in COLUMN_GUESS.items():
            guessed = next((lookup[a] for a in aliases if a in lookup), None)
            options = [NOT_SELECTED] + actual_cols
            default_index = options.index(guessed) if guessed in options else 0
            selection[field] = st.selectbox(
                MAPPING_FIELD_LABELS[field],
                options,
                index=default_index,
                key=f"map_{file_label}_{field}",
            )

    if any(chosen == NOT_SELECTED for chosen in selection.values()):
        st.error("Необходимо выбрать все обязательные колонки")
        st.stop()

    # rename chosen columns to schema fields, then drop everything else
    rename_map = {chosen: field for field, chosen in selection.items()}
    renamed = df.rename(columns=rename_map)
    return renamed[list(COLUMN_GUESS)].copy()


# --- chat tab (group-level data) --------------------------------------------
@st.cache_resource
def init_group_warehouse() -> str:
    """Generate the synthetic group and build its warehouse exactly once.

    Cached across reruns so the SQLite warehouse is built a single time per
    session rather than on every Streamlit rerun. Returns the db path.
    """
    df, _ = generate_group()
    return build_warehouse(df)


def render_chat_tab() -> None:
    """Chat UI over the group warehouse. No math/SQL here — only answer() calls."""
    db_path = init_group_warehouse()

    st.subheader("Чат с данными группы")
    st.caption("Задайте вопрос на русском по закупкам всех дочерних компаний")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if st.button("Очистить чат"):
        st.session_state["chat_history"] = []
        st.rerun()

    # Render history top to bottom.
    for entry in st.session_state["chat_history"]:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            if entry.get("error"):
                st.error(entry["error"])
            else:
                st.write(entry["answer"])
        with st.expander("Запрос к базе"):
            if entry.get("sql"):
                st.code(entry["sql"], language="sql")
            if entry.get("data") is not None:
                st.dataframe(entry["data"], use_container_width=True)

    question = st.chat_input("Спросите про закупки группы…")
    if question:
        res = answer(question, db_path)
        st.session_state["chat_history"].append(
            {
                "question": res["question"],
                "answer": res["answer"],
                "sql": res["sql"],
                "data": res["data"],
                "error": res["error"],
            }
        )
        st.rerun()


def render_analytics_tab() -> None:
    """Group-wide analytics dashboard. All aggregation lives in core.warehouse."""
    db_path = init_group_warehouse()

    st.subheader("Аналитика по группе")
    st.caption("Сводные показатели закупок по всем дочерним компаниям")

    # 1. Headline metric cards.
    summary = get_group_summary(db_path)
    c1, c2, c3 = st.columns(3)
    c1.metric("Общие расходы группы, ₸", MONEY_FMT.format(summary["total_spend"]))
    c2.metric("Дочерних компаний", summary["n_subsidiaries"])
    c3.metric("Лет в данных", summary["n_years"])
    st.divider()

    # 2. Spend per subsidiary, latest year, descending.
    st.subheader(f"Расходы по дочерним компаниям — {LATEST_YEAR}")
    latest = get_subsidiary_totals_for_year(LATEST_YEAR, db_path)
    st.bar_chart(latest, x="subsidiary", y="total_spend")

    # 3. Group spend trend across all years.
    st.subheader("Динамика общих расходов группы по годам")
    yearly = get_yearly_totals(db_path)
    st.line_chart(yearly, x="year", y="total_spend")

    # 4. Ranked subsidiary totals table.
    st.subheader("Рейтинг дочерних компаний по расходам (все годы)")
    totals = get_subsidiary_totals(db_path).rename(
        columns={"rank": "Ранг", "subsidiary": "Дочерняя компания", "total_spend": "Сумма, ₸"}
    )
    st.dataframe(
        totals.style.format({"Сумма, ₸": MONEY_FMT}),
        hide_index=True,
        use_container_width=True,
    )


def render_comparison_tab() -> None:
    """The existing single-sheet comparison workflow, unchanged."""
    # 1. HEADER
    st.title("Инструмент сравнения IT-закупок")
    st.caption("Аудит IT-бюджетов")
    st.divider()

    # 2. CONTROLS
    col_a, col_b, col_t = st.columns([2, 2, 1])
    with col_a:
        file_a = st.file_uploader("Загрузить лист А (xlsx)", type=["xlsx"], key="file_a")
    with col_b:
        file_b = st.file_uploader("Загрузить лист Б (xlsx)", type=["xlsx"], key="file_b")
    with col_t:
        threshold = st.slider("Порог аномалии (%)", min_value=1, max_value=50, value=15, step=1)

    if st.button("Использовать синтетические данные для демо"):
        st.session_state["use_demo"] = True

    # Decide the data source, run the comparison. All math happens inside compare().
    df_a = df_b = None
    using_demo = False
    try:
        if file_a is not None and file_b is not None:
            df_a = normalize(map_columns(pd.read_excel(file_a), "Лист А"))
            df_b = normalize(map_columns(pd.read_excel(file_b), "Лист Б"))
        elif st.session_state.get("use_demo", False):
            df_a, df_b, _ = generate(seed=42)
            using_demo = True
        else:
            st.info("Загрузите два файла или запустите демо")
            st.stop()

        if using_demo:
            st.info("Используются синтетические тестовые данные (seed=42)")

        result = compare(df_a, df_b, threshold_pct=float(threshold))
    except ValueError as e:
        st.error("Ошибка формата файла: " + str(e))
        st.stop()

    summary = result["summary"]
    matched = result["matched"]

    # 3. SUMMARY METRICS
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Общее отклонение", f"{summary['total_pct_delta']:+.1f}%")
    m2.metric("Взвешенное Δ цены", f"{summary['weighted_avg_pct_delta_price']:+.1f}%")
    m3.metric("Доля аномалий", f"{summary['anomaly_rate']:.1f}%")
    with m4:
        sub_matched, sub_only = st.columns(2)
        sub_matched.metric("Совпало", summary["n_matched"])
        sub_only.metric("Только А / Б", f"{summary['n_only_in_a']} / {summary['n_only_in_b']}")
    st.divider()

    # 4. SPLIT-SCREEN
    left, right = st.columns(2)
    with left:
        st.subheader("Лист А — Дочерняя компания А")
        render_sheet(_build_sheet(matched, result["only_in_a"], "a"))
    with right:
        st.subheader("Лист Б — Дочерняя компания Б")
        render_sheet(_build_sheet(matched, result["only_in_b"], "b"))
    st.divider()

    # 5. DELTA TABLE + Pareto
    st.subheader("Таблица расхождений (совпавшие позиции)")
    render_delta_table(matched)
    if len(matched) >= 2:
        st.caption(
            f"Топ-2 позиции объясняют {matched['cumulative_pct'].iloc[1]:.1f}% суммарного отклонения"
        )
    st.divider()

    # 6. MISSING / NEW ROWS
    only_left, only_right = st.columns(2)
    with only_left:
        st.subheader("Только в листе А (отсутствуют в Б)")
        if len(result["only_in_a"]) == 0:
            st.success("Все позиции листа А присутствуют в листе Б")
        else:
            render_only_in(result["only_in_a"], ONLY_A)
    with only_right:
        st.subheader("Только в листе Б (новые позиции)")
        if len(result["only_in_b"]) == 0:
            st.success("Все позиции листа Б присутствуют в листе А")
        else:
            render_only_in(result["only_in_b"], ONLY_B)
    st.divider()

    # 7. ROLLUPS
    with st.expander("Разбивка по поставщикам", expanded=False):
        render_rollup(result["supplier_rollup"], "supplier", "Поставщик")
    with st.expander("Разбивка по категориям", expanded=False):
        render_rollup(result["category_rollup"], "category", "Категория")

    # 8. FOOTER
    st.divider()
    st.caption("Аудит IT-бюджетов · MVP v1.0")


# --- audit tab (corporate ИТ-бюджет ingest + rule audit + AI narrative + exports) ---
@st.cache_data(show_spinner=False)
def _cached_extract_workbook(file_bytes: bytes) -> list[dict]:
    """extract_workbook() over an uploaded file's bytes, cached on those bytes."""
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        return extract_workbook(tmp_path)
    finally:
        os.unlink(tmp_path)


@st.cache_data(show_spinner=False)
def _cached_run_audit(file_bytes: bytes, sheet_name: str, atk_threshold: float, _sheet_result: dict) -> dict:
    """run_full_audit(), cached on the hashable identifying args (not the DataFrame)."""
    return run_full_audit(_sheet_result, atk_threshold=atk_threshold)


@st.cache_data(show_spinner=False)
def _cached_narrate(file_bytes: bytes, sheet_name: str, atk_threshold: float, _audit_result: dict) -> dict | None:
    """narrate_audit(), cached the same way — avoids a Gemini call on every rerun."""
    return narrate_audit(_audit_result)


@st.cache_data(show_spinner=False)
def _cached_conclusion_bytes(
    file_bytes: bytes, sheet_name: str, atk_threshold: float, has_narrative: bool, _audit_result: dict
) -> bytes:
    """generate_conclusion(), cached — the Node subprocess call isn't free."""
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        generate_conclusion(_audit_result, tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp_path)


def _audit_section_table(all_rows: pd.DataFrame, decision: str) -> pd.DataFrame:
    """Reshape all_rows (filtered to one decision) into that section's display columns."""
    subset = all_rows[all_rows["decision"] == decision]
    if subset.empty:
        return subset

    if decision == "согласовано":
        return pd.DataFrame({
            "Контрагент": _safe_text_col(subset, "counterparty"),
            "Статья": _safe_text_col(subset, "article_code"),
            "Текущий год, тыс. ₸": _safe_col(subset, "plan_annual", empty=None),
            "Δ%": _safe_col(subset, "delta_pct", empty=None),
        })

    return pd.DataFrame({
        "Контрагент": _safe_text_col(subset, "counterparty"),
        "Статья": _safe_text_col(subset, "article_code"),
        "Прошлый год, тыс. ₸": _safe_col(subset, "plan_prior", empty=None),
        "Текущий год, тыс. ₸": _safe_col(subset, "plan_annual", empty=None),
        "Δ тыс. ₸": _safe_col(subset, "delta_abs", empty=None),
        "Δ%": _safe_col(subset, "delta_pct", empty=None),
        "Основание": _safe_text_col(subset, "reason"),
    })


def _render_audit_section(decision: str, count: int, amount: float, disp: pd.DataFrame) -> None:
    """One colored card + its data table, per the АТК/уточнение/согласовано palette."""
    bg = AUDIT_SECTION_BG[decision]
    text_color = AUDIT_SECTION_TEXT[decision]
    text_class = AUDIT_SECTION_TEXT_CLASS[decision]
    icon = AUDIT_SECTION_ICON[decision]
    label = AUDIT_SECTION_LABEL[decision]

    st.markdown(
        f'<div style="background:{bg};padding:10px 18px;border-radius:8px 8px 0 0">'
        f'<b class="{text_class}" style="font-size:15px">{icon} {label} — {count} статей · '
        f'{_audit_money(amount)} тыс. ₸</b></div>',
        unsafe_allow_html=True,
    )

    if disp.empty:
        st.caption("Нет статей в этой категории.")
        return

    money_cols = [c for c in disp.columns if "тыс. ₸" in c]
    pct_cols = [c for c in disp.columns if c == "Δ%"]
    fmt = {c: _audit_money for c in money_cols}
    fmt.update({c: _audit_pct for c in pct_cols})

    styled = disp.style.apply(
        lambda row: [f"background-color:{bg};color:{text_color}"] * len(row), axis=1
    ).format(fmt)
    st.dataframe(styled, use_container_width=True, hide_index=True)


def _audit_xlsx_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Аудит")
    return buffer.getvalue()


def render_audit_tab() -> None:
    """Corporate ИТ-бюджет upload -> core.budget_audit rules -> AI narrative -> exports.

    No math/audit logic here — every number comes from core.corporate_ingest,
    core.budget_audit, core.budget_narrator, and core.conclusion_generator.
    This function only reshapes, formats, and renders what those return.
    """
    inject_audit_css()

    # 1. HEADER BLOCK
    st.markdown(
        '<div style="background:#1F3A5F;padding:24px 28px;border-radius:10px;margin-bottom:20px">'
        '<div class="audit-c-white" style="margin:0;font-size:22px;font-weight:700">Аудит IT-бюджетов компании</div>'
        '<p class="audit-c-subtitle" style="margin:6px 0 0;font-size:14px">Загрузите файл формы — система '
        'проверит квартальные расхождения, динамику по годам и концентрацию закупок</p></div>',
        unsafe_allow_html=True,
    )

    # 2. FILE UPLOAD ROW
    col_current, col_prior, col_threshold = st.columns([2, 2, 1])
    with col_current:
        file_current = st.file_uploader(
            "Файл текущего года (xlsx)", type=["xlsx"], key="audit_file_current"
        )
        st.caption("Формат: ИТ-расходы, листы OPEX/CAPEX")
    with col_prior:
        st.file_uploader(
            "Файл прошлого года (необязательно)", type=["xlsx"], key="audit_file_prior"
        )
        st.caption("Для сравнения год к году")
    with col_threshold:
        atk_threshold = st.slider(
            "Порог АТК (%)", min_value=30, max_value=100, value=50, step=5, key="atk_threshold"
        )
        st.caption("Рост выше порога → выносится на АТК")
    # Note: the prior-year file above is accepted but not yet consumed — merging
    # a separate prior-year workbook into the delta computation is a cross-file
    # join core.budget_audit doesn't implement. Delta/АТК comparison currently
    # works from a single file that carries both plan_prior and plan_annual
    # columns itself (the common layout), which is what's wired up below.

    if file_current is None:
        st.info("Загрузите файл формы текущего года, чтобы начать аудит.")
        return

    try:
        file_bytes = file_current.getvalue()

        with st.spinner("Анализируем файл..."):
            sheet_results = _cached_extract_workbook(file_bytes)

        if not sheet_results:
            st.error("Ошибка разбора файла: в книге не найдено ни одного листа.")
            st.stop()

        # 4. SHEET SELECTOR
        sheet_names = [r["sheet_name"] for r in sheet_results]
        if len(sheet_names) > 1:
            selected_sheet = st.selectbox("Выберите лист:", sheet_names, key="audit_sheet")
        else:
            selected_sheet = sheet_names[0]
        sheet_result = next(r for r in sheet_results if r["sheet_name"] == selected_sheet)

        with st.spinner("Анализируем файл..."):
            audit_result = _cached_run_audit(file_bytes, selected_sheet, float(atk_threshold), sheet_result)

        entity = audit_result["entity"]
        budget_year = audit_result["budget_year"]
        summary = audit_result["summary"]

        if not os.environ.get("GEMINI_API_KEY"):
            st.info("AI-объяснения недоступны — задайте GEMINI_API_KEY. Аудит работает полностью без AI.")
            narrative = None
        else:
            with st.spinner("Анализируем файл..."):
                narrated = _cached_narrate(file_bytes, selected_sheet, float(atk_threshold), audit_result)
            narrative = narrated.get("narrative") if narrated else None

        # 5. SUMMARY METRICS BAR
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric(
            "На АТК",
            f"{summary['atk_count']} статей",
            delta=f"−{_audit_money(summary['atk_amount'])} тыс. ₸",
            delta_color="inverse",
        )
        m2.metric("Уточнений", f"{summary['clarification_count']} статей")
        m3.metric("Согласовано", f"{summary['approved_count']} статей", delta_color="normal")
        m4.metric("Итого бюджет", f"{_audit_money(summary['total_amount'])} тыс. ₸")
        with m5:
            risk_level = (narrative or {}).get("risk_level", "")
            risk_label = {"высокий": "Высокий", "средний": "Средний", "низкий": "Низкий"}.get(risk_level, "Н/Д")
            risk_class = RISK_TEXT_CLASS.get(risk_level, "audit-c-risk-unknown")
            st.markdown(
                f'<div class="{risk_class}" style="font-size:28px;font-weight:700">'
                f'⬤ {risk_label}</div>',
                unsafe_allow_html=True,
            )
            st.caption("Уровень риска")

        st.divider()

        # 6. AI SUMMARY
        if narrative and narrative.get("executive_summary"):
            st.markdown(
                '<div style="background:#EFF6FF;padding:16px 20px;border-left:4px solid #1F3A5F;'
                'border-radius:0 8px 8px 0;margin-bottom:16px"><b class="audit-c-navy">Резюме аудита</b>'
                f'<br><span class="audit-c-body">{narrative["executive_summary"]}</span></div>',
                unsafe_allow_html=True,
            )
        elif narrative and narrative.get("error"):
            st.info(f"AI-резюме недоступно: {narrative['error']}")

        if not audit_result["has_prior_year"]:
            st.warning(
                "Файл не содержит данных прошлого года — сравнение год к году недоступно. "
                "Загрузите файл прошлого года для полного анализа."
            )

        # 7. THREE SECTION TABLES
        all_rows = audit_result["all_rows"]
        _render_audit_section("АТК", summary["atk_count"], summary["atk_amount"], _audit_section_table(all_rows, "АТК"))
        _render_audit_section(
            "уточнение", summary["clarification_count"], summary["clarification_amount"],
            _audit_section_table(all_rows, "уточнение"),
        )
        _render_audit_section(
            "согласовано", summary["approved_count"], summary["approved_amount"],
            _audit_section_table(all_rows, "согласовано"),
        )

        # 8. CONCENTRATION CHART
        st.subheader("Концентрация закупок по поставщикам")
        concentration = audit_result["findings"]["concentration"]
        breakdown = concentration["supplier_breakdown"]
        if breakdown is not None and not breakdown.empty:
            st.bar_chart(breakdown.set_index("counterparty")["share_pct"])
        else:
            st.caption("Недостаточно данных для анализа концентрации.")
        if concentration["concentration_flag"]:
            st.markdown(
                '<div style="background:#FFCDD2;padding:12px 18px;border-radius:6px;'
                'border-left:4px solid #C62828"><b class="audit-c-atk-text">Риск концентрации:</b> '
                f'<span class="audit-c-body">{concentration["top_supplier"]} составляет '
                f'{concentration["top_supplier_pct"]:.1f}% бюджета</span></div>',
                unsafe_allow_html=True,
            )

        # 9. QUARTERLY MISMATCHES
        quarterly = audit_result["findings"]["quarterly_mismatches"]
        if quarterly is not None and not quarterly.empty:
            st.subheader("Квартальные расхождения")
            quarterly_disp = pd.DataFrame({
                "Контрагент": _safe_text_col(quarterly, "counterparty"),
                "Статья": _safe_text_col(quarterly, "article_code"),
                "План год, тыс. ₸": quarterly["plan_annual"],
                "Сумма по кварталам, тыс. ₸": quarterly["quarterly_sum"],
                "Расхождение, тыс. ₸": quarterly["difference"],
                "Расхождение, %": quarterly["difference_pct"],
                "Критичность": quarterly["severity"],
            })
            st.dataframe(
                quarterly_disp.style.format({
                    "План год, тыс. ₸": _audit_money,
                    "Сумма по кварталам, тыс. ₸": _audit_money,
                    "Расхождение, тыс. ₸": _audit_money,
                    "Расхождение, %": lambda v: f"{v:.1f}%",
                }),
                use_container_width=True,
                hide_index=True,
            )
            st.bar_chart(quarterly.set_index("counterparty")["difference_pct"])

        st.divider()

        # 10. DOWNLOAD ROW
        dl_left, dl_right = st.columns(2)
        with dl_left:
            with st.spinner("Формируем Заключение..."):
                docx_bytes = _cached_conclusion_bytes(
                    file_bytes, selected_sheet, float(atk_threshold), narrative is not None, audit_result
                )
            st.download_button(
                label="Скачать Заключение (.docx)",
                data=docx_bytes,
                file_name=f"Заключение_{entity}_{budget_year}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                help="Документ готов к редактированию в Word перед подписанием",
            )
        with dl_right:
            st.download_button(
                label="Экспорт данных (.xlsx)",
                data=_audit_xlsx_bytes(all_rows),
                file_name=f"Аудит_{entity}_{budget_year}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        st.caption("Оба документа готовы к редактированию перед подписанием")

        st.divider()
        st.caption("Аудит IT-бюджетов · MVP v2.0")

    except Exception as e:
        st.error("Ошибка разбора файла: " + str(e))
        st.stop()


# --- page --------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Аудит закупок компании", page_icon=None)
inject_css()

# Global header shown above the tabs, so someone opening the app cold understands
# the layout before clicking anything.
st.title("Аудит закупок компании")
st.caption(
    "**Сравнение листов** — сопоставление двух Excel-файлов закупок с поиском "
    "аномалий цен.  ·  **Чат с данными** — вопросы по закупкам всей группы на "
    "русском языке.  ·  **Аналитика по группе** — сводные графики и рейтинг "
    "дочерних компаний."
)

# First-run action: build the synthetic group warehouse explicitly so a new user
# knows exactly what to click before opening the Чат / Аналитика tabs.
if st.button("Загрузить демо-данные", type="primary"):
    init_group_warehouse()
    st.session_state["group_loaded"] = True
if st.session_state.get("group_loaded"):
    st.success(
        "Демо-данные группы загружены — вкладки «Чат с данными» и "
        "«Аналитика по группе» готовы к работе."
    )
st.divider()

tab_compare, tab_chat, tab_analytics, tab_audit = st.tabs(
    ["Сравнение листов", "Чат с данными", "Аналитика по группе", "Аудит бюджета"]
)

# st.tabs() doesn't give each tab its own execution scope — every `with tab_x:`
# block runs in one single top-to-bottom script pass, so any st.stop() call
# anywhere below halts rendering for every tab after it, active or not. Chat
# and analytics never call st.stop(), so they're safe to draw first; compare
# and audit both do (compare on no-file-uploaded, audit on a parse error), so
# whichever of those two is drawn last would never render if the other one
# stops first. Audit is drawn before compare here so a bare page load (no
# files anywhere yet) still shows the audit tab's own upload prompt rather
# than silently going blank.
with tab_chat:
    render_chat_tab()

with tab_analytics:
    render_analytics_tab()

with tab_audit:
    render_audit_tab()

with tab_compare:
    render_comparison_tab()
