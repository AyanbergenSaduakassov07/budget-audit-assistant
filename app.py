"""Streamlit UI for the procurement comparison tool.

This is the ONLY file allowed to import streamlit (SPEC.md architecture rule).
It contains no arithmetic or comparison logic — every number shown here comes
from core.compare / core.schema. app.py only reshapes, renames, formats, and
renders what core returns.
"""

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

tab_compare, tab_chat, tab_analytics = st.tabs(
    ["Сравнение листов", "Чат с данными", "Аналитика по группе"]
)

# Chat and analytics tabs are drawn before the comparison tab because the
# comparison flow calls st.stop() on missing input, which halts the whole script.
with tab_chat:
    render_chat_tab()

with tab_analytics:
    render_analytics_tab()

with tab_compare:
    render_comparison_tab()
