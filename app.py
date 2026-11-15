"""IT Budget Auditor — single-page, chat-first Streamlit app.

The conversation IS the app (ChatGPT-style): one st.chat_input with
accept_file=True is the only input on the page. Dropping a ИТ-бюджет .xlsx into it
runs the existing parse -> flag_rows -> aggregate_workbook pipeline and pushes
the full analysis as ONE assistant message (summary text, CAPEX/OPEX table +
chart, year trend, three collapsed flag expanders). Every follow-up question
goes through the same box and is answered by core chat_answer() over the whole
file.

Streamlit imports live only in app.py (SPEC.md architecture rule).
All parsing/analysis math lives in core.budget_auditor.{parser,analysis};
app.py only reshapes, renames, formats, and renders what that layer returns.

Layout: centered ~820px column, quiet single-line title top-left, bordered
chat bubbles on a warm near-white surface, navy #1F3A5F as accent only.
The YoY threshold slider lives in a "Настройки" popover pinned near the
chat input.

State model (st.session_state):
  file_chat_history — ordered list of messages. Text entries keep the original
      {"role", "content"} shape; a file drop appends an assistant entry with
      kind="analysis" carrying the workbook bytes, re-rendered on every rerun
      through the cached parser.
  active_file — {"name", "bytes"} of the most recent upload; chat_answer()
      always runs against it.
  yoy_threshold — slider value; flags recompute live on change.
"""

import logging
import os
import tempfile
import uuid

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
from core.budget_auditor.narrator_bridge import build_audit_result, narrative_opening
from core.budget_auditor.file_sql import answer_over_file
from core.conclusion_generator import generate_conclusion
from core.conclusion_pptx_generator import generate_pptx_conclusion

logger = logging.getLogger(__name__)

# --- palette (SPEC design system — navy is an accent, never a background fill)
NAVY = "#1F3A5F"
SURFACE = "#FAF9F6"
BUBBLE = "#FFFFFF"
BORDER = "#E3E0D8"
INK = "#2C2C2A"
INK_MUTED = "#6B6960"
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

# The quarterly-mismatch table shows reconciliation evidence instead of the
# YoY delta/justification columns.
QUARTER_VIEW_LABELS = {
    "branch": "Подразделение",
    "cost_type": "Тип",
    "statya_code": "Статья",
    "item_name": "Позиция",
    "counterparty": "Контрагент",
    "quarterly_sum": "Σ кварталов",
    "quarterly_gap_abs": "Разница",
    "quarterly_gap_pct": "Разница %",
    "quarterly_severity": "Серьёзность",
}
SEVERITY_RU = {"critical": "критично", "minor": "незначительно"}

MONEY_FMT = "{:,.0f}"

NO_FILE_REPLY = (
    "Сначала загрузите файл формы ИТ-расходов — прикрепите .xlsx "
    "через скрепку в поле ввода."
)
FALLBACK_REPLY = (
    "Не нашёл точного ответа на этот вопрос. Попробуйте указать год "
    "(например, «план на 2026») или ключевое слово статьи расхода."
)


# --- rendering helpers (presentation only, no math) ---------------------------
def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        /* Centered reading column, 8px spacing rhythm */
        [data-testid="stMainBlockContainer"], .main .block-container {{
            max-width: 820px; margin: 0 auto; padding: 64px 24px 8px;
        }}
        [data-testid="stBottomBlockContainer"], [data-testid="stBottom"] > div {{
            max-width: 820px; margin: 0 auto;
        }}
        .stApp, [data-testid="stBottom"] {{ background: {SURFACE}; }}
        [data-testid="stHeader"] {{ background: transparent; }}

        /* Type: 16px body, 1.5 line-height, ink on warm white */
        .stMarkdown p, .stMarkdown li, .stChatMessage p {{
            font-size: 16px; line-height: 1.5; color: {INK};
        }}
        h1, h2, h3, h4, h5, h6 {{ color: {NAVY} !important; }}
        .stCaption, [data-testid="stCaptionContainer"] p {{ color: {INK_MUTED} !important; }}
        a {{ color: {NAVY}; }}

        .app-title {{
            font-size: 15px; font-weight: 600; color: {NAVY};
            margin: 0 0 16px; letter-spacing: .01em;
        }}

        /* Chat bubbles: subtle border, no heavy fill */
        [data-testid="stChatMessage"] {{
            background: {BUBBLE}; border: 1px solid {BORDER};
            border-radius: 12px; padding: 16px; margin-bottom: 8px;
        }}

        /* Chat input: quiet border, navy only on focus */
        [data-testid="stChatInput"] {{
            background: {BUBBLE}; border: 1px solid {BORDER}; border-radius: 12px;
        }}
        [data-testid="stChatInput"]:focus-within {{
            border-color: {NAVY}; box-shadow: 0 0 0 1px {NAVY}33;
        }}

        /* Buttons: navy accent (outline -> filled on hover), 150ms */
        .stButton > button, [data-testid="stPopover"] button {{
            color: {NAVY}; background: {BUBBLE}; border: 1px solid {NAVY};
            border-radius: 8px; min-height: 40px; transition: background 150ms ease-out;
        }}
        .stButton > button:hover, [data-testid="stPopover"] button:hover {{
            background: {NAVY}; color: #FFFFFF; border-color: {NAVY};
        }}
        @media (prefers-reduced-motion: reduce) {{
            .stButton > button, [data-testid="stPopover"] button {{ transition: none; }}
        }}

        /* Settings popover pinned near the chat input (bottom-right of column).
           z-index must beat stBottom's layer or the trigger is unclickable;
           width must collapse to content or it stretches into a full-width bar. */
        [data-testid="stPopover"] {{
            position: fixed; bottom: 96px; z-index: 1000;
            right: max(24px, calc((100vw - 868px) / 2));
            left: auto; width: auto;
        }}
        [data-testid="stPopover"] > button {{ width: auto; }}

        /* Avatars: assistant in accent navy, user in neutral ink — never the
           loud red/orange defaults (they'd be the strongest color on the page) */
        [data-testid="chatAvatarIcon-assistant"],
        [data-testid="stChatMessageAvatarAssistant"] {{
            background-color: {NAVY} !important; color: #FFFFFF !important;
        }}
        [data-testid="chatAvatarIcon-user"],
        [data-testid="stChatMessageAvatarUser"] {{
            background-color: {INK_MUTED} !important; color: #FFFFFF !important;
        }}

        [data-testid="stExpander"] {{
            background: {BUBBLE}; border: 1px solid {BORDER}; border-radius: 8px;
        }}
        [data-testid="stDataFrame"] {{ background: {BUBBLE}; border-radius: 6px; }}

        /* Streamlit fades already-rendered content while a rerun is in
           flight (its "stale element" indicator) -- reads as the whole
           page going dim/grey mid-demo, which looks broken to someone who
           doesn't know Streamlit's internals. Force full opacity throughout
           a run. Not a single stable documented selector for this across
           versions, so several overlapping ones are forced together;
           harmless if some don't match this Streamlit build. Cache-spinner
           text (e.g. "Формирую заключение…") is untouched -- that's a
           separate, wanted signal that something is happening. */
        [data-testid="stAppViewContainer"],
        [data-testid="stMainBlockContainer"],
        [data-testid="stBottomBlockContainer"],
        [data-testid="stVerticalBlock"],
        [data-testid="stVerticalBlockBorderWrapper"],
        .main .block-container,
        .element-container,
        .stChatMessage {{
            opacity: 1 !important;
            transition: none !important;
        }}

        /* Empty state: vertically centered short prompt */
        .empty-state {{
            min-height: 55vh; display: flex; flex-direction: column;
            align-items: center; justify-content: center; text-align: center; gap: 8px;
        }}
        .empty-state .es-title {{ font-size: 20px; font-weight: 600; color: {NAVY}; margin: 0; }}
        .empty-state .es-sub {{ font-size: 15px; color: {INK_MUTED}; max-width: 60ch; margin: 0; line-height: 1.5; }}
        .empty-state .es-hint {{ font-size: 13px; color: {INK_MUTED}; margin-top: 16px; }}
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


def _format_eager(disp: pd.DataFrame, fmt: dict) -> pd.DataFrame:
    """Apply formatters as values, not via Styler: st.dataframe draws null cells
    as gray 'None' regardless of Styler display text, so NA must become 'н/д'
    before the frame reaches the grid.

    Columns left out of fmt (e.g. Обоснование, Контрагент) are raw passthrough
    from the source Excel and can mix strings with a stray numeric cell (an
    ИНН typed into a counterparty field, say) — object dtype hides this until
    st.dataframe's Arrow conversion raises "Expected bytes, got a 'int'
    object". Coerce those to string too, leaving real null values alone so
    the caller's later .where(notna, "—") still catches them."""
    out = disp.copy()
    for col, formatter in fmt.items():
        if col in out.columns:
            out[col] = out[col].map(formatter)
    for col in out.columns:
        if col not in fmt and out[col].dtype == object:
            out[col] = out[col].map(lambda v: v if pd.isna(v) else str(v))
    return out


def _severity(value) -> str:
    if value is None or pd.isna(value):
        return "н/д"
    return SEVERITY_RU.get(value, str(value))


def build_quarterly_view(subset: pd.DataFrame, plan_col: str | None) -> tuple[pd.DataFrame, dict]:
    """Same shape as build_flag_view, for the quarterly-reconciliation table:
    display columns, Russian headers, and the format map."""
    labels = dict(QUARTER_VIEW_LABELS)
    if plan_col:
        labels[plan_col] = f"План {plan_col.removeprefix('plan_')}"
    show = [c for c in
            ["branch", "cost_type", "statya_code", "item_name", "counterparty",
             plan_col, "quarterly_sum", "quarterly_gap_abs", "quarterly_gap_pct",
             "quarterly_severity"]
            if c and c in subset.columns]
    disp = subset[show].rename(columns=labels).reset_index(drop=True)
    fmt = {}
    if plan_col and labels[plan_col] in disp.columns:
        fmt[labels[plan_col]] = _money
    fmt.update({"Σ кварталов": _money, "Разница": _money,
                "Разница %": _share, "Серьёзность": _severity})
    return disp, fmt


def render_flag_view(subset: pd.DataFrame, plan_col: str | None) -> None:
    disp, fmt = build_flag_view(subset, plan_col)
    out = _format_eager(disp, fmt)
    # Unformatted object columns (Обоснование, Контрагент) can hold raw None —
    # show an em dash instead of the grid's gray "None".
    st.dataframe(out.where(out.notna(), "—"), use_container_width=True, hide_index=True)


def _analyze(file_bytes: bytes, threshold: float) -> dict:
    """Existing pipeline, unchanged: parse -> flag_rows -> aggregate_workbook.

    Returns everything a render pass or chat_answer() needs. Raises whatever
    parse_workbook raises on an unreadable file.
    """
    meta, sheets = _cached_parse(file_bytes)
    usable = {sn: d for sn, d in sheets.items() if d["df"] is not None and not d["df"].empty}
    flagged = {
        sn: {"df": flag_rows(d["df"], d["stats"], threshold_pct=threshold), "stats": d["stats"]}
        for sn, d in usable.items()
    }
    combined = aggregate_workbook(flagged)
    plan_year = (
        combined["plan_year"].dropna().iloc[0]
        if not combined.empty and combined["plan_year"].notna().any()
        else None
    )
    return {
        "meta": meta,
        "flagged_sheets": flagged,
        "combined": combined,
        "empty_sheets": [sn for sn in sheets if sn not in usable],
        "plan_year": plan_year,
        "plan_col": f"plan_{plan_year}" if plan_year else None,
        # Sheets whose column layout came from the LLM fallback (parser.py's
        # deterministic pass couldn't find/trust a header on them) rather
        # than the validated template rules -- see render_analysis.
        "llm_fallback_sheets": [sn for sn, d in flagged.items() if d["stats"].get("parsed_via") == "llm_fallback"],
    }


def _current_threshold() -> float:
    return float(st.session_state.get("yoy_threshold", DEFAULT_THRESHOLD))


DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"


@st.cache_data(show_spinner=False)
def _cached_conclusion_docx(file_bytes: bytes, threshold: float) -> tuple[bytes, str] | None:
    """Заключение .docx (bytes, filename), cached per (file, threshold) so the
    Node render runs once, not on every rerun. None on any failure (e.g. node
    not installed) — the download button is simply not rendered."""
    try:
        a = _analyze(file_bytes, threshold)
        result = build_audit_result(a["meta"], a["combined"])
        entity = str(result.get("entity") or "Компания").strip().replace(" ", "_")
        filename = f"Заключение_{entity}_{result.get('budget_year') or ''}.docx".replace("_.", ".")
        out_dir = tempfile.mkdtemp()
        out_path = generate_conclusion(result, os.path.join(out_dir, filename))
        with open(out_path, "rb") as fh:
            data = fh.read()
        os.unlink(out_path)
        os.rmdir(out_dir)
        return data, filename
    except Exception:
        logger.exception("conclusion docx generation failed; button not shown")
        return None


@st.cache_data(show_spinner=False)
def _cached_conclusion_pptx(file_bytes: bytes, threshold: float) -> tuple[bytes, str] | None:
    """Заключение .pptx (bytes, filename), cached per (file, threshold) — same
    pattern as _cached_conclusion_docx. None on any failure (e.g. a generation
    error) — the download button is simply not rendered."""
    try:
        a = _analyze(file_bytes, threshold)
        result = build_audit_result(a["meta"], a["combined"])
        entity = str(result.get("entity") or "Компания").strip().replace(" ", "_")
        filename = f"Заключение_{entity}_{result.get('budget_year') or ''}.pptx".replace("_.", ".")
        out_dir = tempfile.mkdtemp()
        out_path = generate_pptx_conclusion(
            result, os.path.join(out_dir, filename), threshold_pct=threshold,
        )
        with open(out_path, "rb") as fh:
            data = fh.read()
        os.unlink(out_path)
        os.rmdir(out_dir)
        return data, filename
    except Exception:
        logger.exception("conclusion pptx generation failed; button not shown")
        return None


@st.cache_data(show_spinner="Формирую заключение…")
def _cached_opening(file_bytes: bytes, threshold: float) -> str | None:
    """LLM executive summary for the analysis opening, cached per
    (file, threshold) so history re-renders never repeat the API call.
    None (no key / any failure) means: use the template summary."""
    a = _analyze(file_bytes, threshold)
    return narrative_opening(a["meta"], a["combined"])


def render_capex_opex(a: dict) -> None:
    st.markdown(f"**CAPEX vs OPEX — {a['plan_year'] if a['plan_year'] else 'год кампании'}**")
    cvo = capex_opex_table(a["combined"])
    if cvo.empty:
        st.caption("Недостаточно данных для сравнения CAPEX/OPEX.")
        return
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
    st.dataframe(_format_eager(cvo, fmt), use_container_width=True, hide_index=True)
    if "Δ %" not in cvo.columns:
        st.caption(
            "Δ год-к-году не показана: в форме нет колонок прошлого года "
            "(структурный пробел формы, не ошибка загрузки)."
        )
    if plan_label:
        st.bar_chart(cvo.set_index("Тип")[plan_label], color=NAVY, height=180, horizontal=True)


def render_trend(a: dict) -> None:
    trend = year_trend(a["combined"])
    if trend.empty:
        return
    st.markdown("**Динамика по годам**")
    # str years so the axis doesn't render 2026 as "2,026"
    wide = (
        trend.assign(Год=trend["Год"].astype(str))
        .pivot_table(index="Год", columns="Тип", values="Сумма", aggfunc="sum")
    )
    st.line_chart(wide, color=[TREND_COLORS.get(c, NAVY) for c in wide.columns], height=240)
    trend_note = year_trend_text(a["combined"])
    if trend["unit_mismatch_suspected"].any():
        st.warning(trend_note)
    else:
        st.caption(trend_note)


def render_flags(a: dict, threshold: float) -> None:
    """Three flag categories: collapsed expander when non-empty, ✓ line when clean."""
    st.markdown("**Проблемные и рискованные статьи**")
    combined, flagged = a["combined"], a["flagged_sheets"]
    no_prior = [sn for sn, d in flagged.items()
                if _best_baseline_col(d["df"], d["stats"].get("plan_year")) is None]
    no_just = [sn for sn, d in flagged.items() if not d["stats"]["has_justification"]]

    def subset(flag: str) -> pd.DataFrame:
        return combined[combined[flag]] if flag in combined else pd.DataFrame()

    d = subset("high_change_flag")
    if d.empty:
        st.caption(f"✓ Резких изменений год-к-году (порог {threshold:.0f}%) не найдено.")
    else:
        with st.expander(f"Резкое изменение г/г — {len(d)} (порог {threshold:.0f}%)"):
            if no_prior:
                st.caption("Листы без данных прошлого года (сравнение г/г структурно "
                           "невозможно): " + ", ".join(no_prior))
            render_flag_view(d, a["plan_col"])
    if no_prior and d.empty:
        st.caption("Листы без данных прошлого года: " + ", ".join(no_prior))

    d = subset("missing_justification_flag")
    if d.empty:
        st.caption("✓ Все статьи имеют заполненное обоснование.")
    else:
        with st.expander(f"Без обоснования — {len(d)}"):
            if no_just:
                st.warning(
                    "В этих листах колонка «Обоснование» отсутствует целиком — все их "
                    "статьи считаются необоснованными: " + ", ".join(no_just)
                )
            render_flag_view(d, a["plan_col"])

    d = subset("high_price_flag")
    if d.empty:
        st.caption("✓ Ценовых выбросов не найдено.")
    else:
        with st.expander(f"Топ-10% по стоимости — {len(d)} (внутри своего листа)"):
            render_flag_view(d, a["plan_col"])

    d = subset("quarterly_mismatch_flag")
    no_quarters = [sn for sn, dd in flagged.items()
                   if not any(c.startswith("plan_") and "_q" in c for c in dd["df"].columns)]
    if d.empty:
        if len(no_quarters) == len(flagged):
            st.caption("Квартальная разбивка в форме отсутствует — сверка кварталов невозможна.")
        else:
            st.caption("✓ Квартальные суммы согласуются с годовым планом (допуск 1%).")
            if no_quarters:
                st.caption("Листы без квартальной разбивки: " + ", ".join(no_quarters))
    else:
        n_critical = int((d["quarterly_severity"] == "critical").sum())
        label = f"Несоответствие по кварталам — {len(d)}"
        if n_critical:
            label += f" (критичных: {n_critical})"
        with st.expander(label):
            st.caption("Сумма кварталов расходится с годовым планом более чем на 1%; "
                       "свыше 5% — критично.")
            if no_quarters:
                st.caption("Листы без квартальной разбивки (не проверялись): "
                           + ", ".join(no_quarters))
            disp, fmt = build_quarterly_view(d, a["plan_col"])
            out = _format_eager(disp, fmt)
            st.dataframe(out.where(out.notna(), "—"), use_container_width=True, hide_index=True)

    d = subset("anomalous_code_flag")
    if d.empty:
        st.caption("✓ Все коды статей соответствуют формату (например IT-CA-01).")
    else:
        with st.expander(f"Нестандартный формат кода статьи — {len(d)}"):
            st.caption(
                "Код статьи не соответствует ожидаемому формату (например IT-CA-01) — "
                "предупреждение о качестве данных, возможна опечатка или нестандартная "
                "нумерация. Не влияет на решение по статье."
            )
            render_flag_view(d, a["plan_col"])


def render_analysis(entry: dict) -> None:
    """The full file analysis as ONE assistant message body."""
    try:
        a = _analyze(entry["bytes"], _current_threshold())
    except Exception as e:
        logger.exception("workbook parse failed")
        st.error(f"Не удалось разобрать файл «{entry['name']}»: {e}")
        return
    if not a["flagged_sheets"]:
        st.warning(
            "Ни в одном листе не найдено заполненных статей расхода. "
            "Проверьте, что это действительно форма ИТ-расходов."
        )
        return
    # Opening text: LLM executive summary when OPENROUTER_API_KEY is set and the
    # call succeeds; otherwise the template summary — silently, never an error.
    opening = _cached_opening(entry["bytes"], _current_threshold())
    st.write(opening or workbook_summary_text(a["meta"], a["flagged_sheets"]))
    if a["empty_sheets"]:
        st.caption(
            "Пустые/незаполненные листы (только шаблон, без данных): "
            + ", ".join(a["empty_sheets"])
        )
    if a["llm_fallback_sheets"]:
        st.warning(
            "Обнаружен нестандартный формат — колонки определены автоматически, "
            "проверьте результат перед использованием (листы: "
            + ", ".join(a["llm_fallback_sheets"]) + ")."
        )
    render_capex_opex(a)
    render_trend(a)
    render_flags(a, _current_threshold())
    docx = _cached_conclusion_docx(entry["bytes"], _current_threshold())
    if docx:
        data, filename = docx
        st.download_button(
            "Скачать заключение (.docx)",
            data=data,
            file_name=filename,
            mime=DOCX_MIME,
            key=f"docx_{entry.get('uid', id(entry))}",
            icon=":material/description:",
        )
    pptx = _cached_conclusion_pptx(entry["bytes"], _current_threshold())
    if pptx:
        data, filename = pptx
        st.download_button(
            "Скачать презентацию (.pptx)",
            data=data,
            file_name=filename,
            mime=PPTX_MIME,
            key=f"pptx_{entry.get('uid', id(entry))}",
            icon=":material/slideshow:",
        )
    st.caption("Задайте вопрос по файлу — например, «сколько CAPEX в 2027» или "
               "«какая ветка тратит больше».")


_MAX_HISTORY_EXCHANGES = 3


def _recent_history_exchanges(n: int = _MAX_HISTORY_EXCHANGES) -> list[tuple[str, str]]:
    """Last n (question, answer) text pairs from file_chat_history, in order.

    Only plain text user/assistant entries are paired — a file-drop entry
    (kind="analysis", no "content" field) breaks any pending pair rather than
    being included, since there's no textual Q&A to hand to the model there.
    Already-rendered strings, no re-parsing; feeds only the AI SQL fallback
    (answer_over_file), not the stateless rule-based chat_answer().
    """
    pairs: list[tuple[str, str]] = []
    pending_question: str | None = None
    for entry in st.session_state.get("file_chat_history", []):
        if entry.get("kind") == "analysis":
            pending_question = None
            continue
        if entry["role"] == "user":
            pending_question = entry.get("content")
        elif entry["role"] == "assistant" and pending_question is not None:
            pairs.append((pending_question, entry.get("content", "")))
            pending_question = None
    return pairs[-n:]


def answer_question(question: str) -> str:
    """Route a chat question to core chat_answer() over the active file."""
    active = st.session_state.get("active_file")
    if active is None:
        return NO_FILE_REPLY
    try:
        a = _analyze(active["bytes"], _current_threshold())
    except Exception:
        logger.exception("workbook parse failed during chat")
        return f"Не удалось разобрать файл «{active['name']}» — загрузите его заново."
    reply = chat_answer(question, a["combined"], a["meta"])
    if reply is not None:
        return reply
    # Rule-based path found nothing: try LLM (OpenRouter) text-to-SQL over
    # this file's data, with recent conversation history for follow-ups like
    # "а за все года". Returns None (silently) if no key / quota / failure ->
    # not-found msg.
    history = _recent_history_exchanges()
    sql_reply = answer_over_file(question, a["combined"], a["meta"], history=history)
    return sql_reply if sql_reply is not None else FALLBACK_REPLY


# --- page ----------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Аудит IT-бюджетов", page_icon=None)
inject_css()

st.markdown('<p class="app-title">Аудит IT-бюджетов</p>', unsafe_allow_html=True)

history = st.session_state.setdefault("file_chat_history", [])

# Empty state: vertically centered prompt; the chat input below is the only form.
if not history:
    st.markdown(
        '<div class="empty-state">'
        '<p class="es-title">Загрузите форму ИТ-расходов</p>'
        '<p class="es-sub">Один файл .xlsx — система опишет его текстом, сравнит '
        'CAPEX и OPEX, покажет динамику по годам, найдёт рискованные статьи '
        'и ответит на вопросы по всему файлу.</p>'
        '<p class="es-hint">Прикрепите файл скрепкой в поле ввода ниже — '
        'или сначала спросите, как это работает.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

for entry in history:
    with st.chat_message(entry["role"]):
        if entry.get("kind") == "analysis":
            render_analysis(entry)
        else:
            st.write(entry["content"])

with st.popover("Настройки", icon=":material/settings:"):
    st.slider(
        "Порог резкого изменения год-к-году, %",
        min_value=5, max_value=50, value=DEFAULT_THRESHOLD, step=1,
        key="yoy_threshold",
    )
    st.caption("Порог применяется ко всем показанным анализам сразу.")
    if history and st.button("Очистить диалог"):
        st.session_state["file_chat_history"] = []
        st.session_state.pop("active_file", None)
        st.rerun()
    st.caption("Аудит IT-бюджетов · прототип")

submitted = st.chat_input(
    "Загрузите файл или задайте вопрос",
    accept_file=True,
    file_type=["xlsx"],
)
if submitted:
    files = submitted.files or []
    text = (submitted.text or "").strip()
    if files:
        f = files[0]
        st.session_state["active_file"] = {"name": f.name, "bytes": f.getvalue()}
        history.append({"role": "user", "content": f"Файл: {f.name}"})
        # uid keeps the download-button widget key unique even when the same
        # file is uploaded twice in one session.
        history.append({"role": "assistant", "kind": "analysis", "name": f.name,
                        "bytes": f.getvalue(), "uid": uuid.uuid4().hex})
    if text:
        history.append({"role": "user", "content": text})
        with st.chat_message("assistant"):
            with st.spinner("Обрабатываю запрос..."):
                reply = answer_question(text)
        history.append({"role": "assistant", "content": reply})
    st.rerun()
