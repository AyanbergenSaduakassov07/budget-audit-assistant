"""Render a build_audit_result() dict as a Заключение по результатам
экспертизы .pptx — a slide-deck companion to conclusion_generator's .docx,
built from the exact same audit_result shape (no second data-mapping path).

Row/summary selection is reused from conclusion_generator wherever it
already exists there (_atk_rows, _clarification_rows, _summary_rows,
_format_amount) rather than re-derived here. CAPEX/OPEX and year-trend
chart data come from core.budget_auditor.parser's own capex_opex_table()/
year_trend() — called either on a caller-supplied table or recomputed from
audit_result["all_rows"] (which carries the same columns as the combined
DataFrame those functions expect; see narrator_bridge._decorate_rows). The
flag-count wording on the risk-overview slide and the year-trend Δ%
mirror app.py's render_flags() and analysis.py's _yoy_delta_str()
respectively, again reused rather than re-derived.

Slide order: Title -> Executive Summary -> Резюме (KPI) -> Проблемные и
рискованные статьи (risk overview) -> CAPEX vs OPEX -> Динамика по годам ->
Концентрация поставщиков (conditional) -> top-risk item tables (АТК /
уточнение, each its own slide, conditional).

Every chart and table is a native pptx object (CategoryChartData-backed
chart, a real table), never a flattened image — a reviewer must be able
to hand-edit this before sign-off, same as the .docx.

Per SPEC.md architecture rule: this module is pure — no streamlit imports.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from core.budget_auditor.analysis import _yoy_delta_str
from core.budget_auditor.narrator_bridge import _bool_col
from core.budget_auditor.parser import capex_opex_table, year_trend
from core.conclusion_generator import _atk_rows, _clarification_rows, _format_amount, _summary_rows

# --- palette (same SPEC design system as app.py / conclusion_generator's .docx)
_NAVY = RGBColor.from_string("1F3A5F")
_SURFACE = RGBColor.from_string("FAF9F6")
_INK = RGBColor.from_string("2C2C2A")
_INK_MUTED = RGBColor.from_string("6B6960")
_WHITE = RGBColor.from_string("FFFFFF")
# Same per-series colors as app.py's TREND_COLORS, duplicated as plain
# constants here rather than imported — importing app.py would run its
# top-level Streamlit page code, which core/ must never depend on.
_TREND_COLORS = {"план": "1F3A5F", "факт": "2E7D32", "прогноз": "F59E0B"}
# Same default as app.py's DEFAULT_THRESHOLD, duplicated for the same reason.
_DEFAULT_THRESHOLD_PCT = 15.0
# A concentration finding worth its own slide: either budget_auditor's own
# >40% concentration_flag, or a share still material enough to be worth
# surfacing (>=25%) even if it didn't cross that flag's threshold.
_CONCENTRATION_SLIDE_MIN_PCT = 25.0

_SLIDE_W = Inches(13.333)
_SLIDE_H = Inches(7.5)
_MAX_RISK_ROWS = 10


def _blank_slide(prs: Presentation):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # "Blank" in the default template
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = _SURFACE
    return slide


def _add_title(slide, text: str, top=Inches(0.4), size=32) -> None:
    box = slide.shapes.add_textbox(Inches(0.6), top, _SLIDE_W - Inches(1.2), Inches(1.0))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = True
    run.font.color.rgb = _NAVY


def _add_body_text(slide, text: str, left, top, width, height, size=16, color=_INK, bold=False, align=None) -> None:
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    if align is not None:
        p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color


def _style_table_header(cell, text: str) -> None:
    cell.text = text
    cell.fill.solid()
    cell.fill.fore_color.rgb = _NAVY
    para = cell.text_frame.paragraphs[0]
    para.font.bold = True
    para.font.color.rgb = _WHITE
    para.font.size = Pt(14)


def _style_table_body_cell(cell, value) -> None:
    cell.text = str(value)
    cell.fill.solid()
    cell.fill.fore_color.rgb = _WHITE
    para = cell.text_frame.paragraphs[0]
    para.font.size = Pt(12)
    para.font.color.rgb = _INK


# --- slide 1: title -------------------------------------------------------------------
def _add_title_slide(prs: Presentation, audit_result: dict, risk_level: str) -> None:
    slide = _blank_slide(prs)
    entity = audit_result.get("entity") or "—"
    budget_year = audit_result.get("budget_year")

    _add_body_text(slide, "Заключение по результатам экспертизы", Inches(0.8), Inches(2.3),
                    _SLIDE_W - Inches(1.6), Inches(1.0), size=30, color=_NAVY, bold=True,
                    align=PP_ALIGN.CENTER)
    _add_body_text(slide, entity, Inches(0.8), Inches(3.4), _SLIDE_W - Inches(1.6), Inches(0.7),
                    size=22, color=_INK, align=PP_ALIGN.CENTER)
    year_line = f"Бюджетный год: {budget_year}" if budget_year else "Бюджетный год: —"
    _add_body_text(slide, year_line, Inches(0.8), Inches(4.1), _SLIDE_W - Inches(1.6), Inches(0.5),
                    size=16, color=_INK_MUTED, align=PP_ALIGN.CENTER)
    if risk_level:
        _add_body_text(slide, f"Уровень риска: {risk_level}", Inches(0.8), Inches(4.6),
                        _SLIDE_W - Inches(1.6), Inches(0.5), size=16, color=_INK_MUTED,
                        align=PP_ALIGN.CENTER)


# --- slide 2: executive summary ---------------------------------------------------------
_NO_JUSTIFICATION_COLUMN_CAPTION = (
    "0 статей отмечено как согласовано, поскольку форма не содержит колонку "
    "«Обоснование» — это структурный пробел формы, а не результат построчной "
    "проверки."
)


def _zero_approved_is_structural_gap(audit_result: dict) -> bool:
    """True only when 0 согласовано is explained by the form itself never
    having an «Обоснование» column — never for a file that genuinely
    approved nothing for some other, real reason."""
    summary = audit_result.get("summary", {}) or {}
    if summary.get("approved_count") != 0:
        return False
    all_rows = audit_result.get("all_rows")
    if all_rows is None:
        all_rows = pd.DataFrame()
    return "justification" not in all_rows.columns


def _add_executive_summary_slide(prs: Presentation, audit_result: dict, executive_summary: str) -> None:
    if not executive_summary:
        return
    slide = _blank_slide(prs)
    _add_title(slide, "Исполнительное резюме")
    _add_body_text(slide, executive_summary, Inches(0.8), Inches(1.6), _SLIDE_W - Inches(1.6),
                    Inches(4.2), size=18, color=_INK)

    if _zero_approved_is_structural_gap(audit_result):
        _add_body_text(slide, _NO_JUSTIFICATION_COLUMN_CAPTION, Inches(0.8), Inches(6.0),
                        _SLIDE_W - Inches(1.6), Inches(1.0), size=13, color=_INK_MUTED)


# --- slide 3: KPI summary (Резюме) ------------------------------------------------------
def _add_summary_slide(prs: Presentation, summary: dict) -> None:
    slide = _blank_slide(prs)
    _add_title(slide, "Резюме")

    rows = _summary_rows(summary)
    buckets = [r for r in rows if not r["bold"]]  # АТК / уточнение / согласовано
    total = next((r for r in rows if r["bold"]), None)  # ИТОГО

    col_w = (_SLIDE_W - Inches(1.2)) / max(len(buckets), 1)
    for i, bucket in enumerate(buckets):
        left = Inches(0.6) + col_w * i
        _add_body_text(slide, bucket["decision"], left, Inches(1.8), col_w, Inches(0.6),
                        size=18, color=_INK_MUTED, align=PP_ALIGN.CENTER)
        _add_body_text(slide, str(bucket["count"]), left, Inches(2.4), col_w, Inches(1.1),
                        size=54, color=_NAVY, bold=True, align=PP_ALIGN.CENTER)
        _add_body_text(slide, f'{bucket["amount"]} тыс. тг', left, Inches(3.5), col_w, Inches(0.6),
                        size=16, color=_INK, align=PP_ALIGN.CENTER)

    if total is not None:
        _add_body_text(slide, f'ИТОГО: {total["count"]} статей — {total["amount"]} тыс. тг',
                        Inches(0.6), Inches(5.2), _SLIDE_W - Inches(1.2), Inches(0.6),
                        size=20, color=_INK, bold=True, align=PP_ALIGN.CENTER)


# --- slide 4: risk overview (Проблемные и рискованные статьи) ---------------------------
# Wording mirrors app.py's render_flags() exactly: a "✓ ..." clean-result
# message per flag when its count is zero, a plain count + short label
# otherwise. "✓" (U+2713, plain check mark) is used rather than an emoji
# glyph, per the same house style as the rest of this deck.
def _add_risk_overview_slide(prs: Presentation, audit_result: dict, threshold_pct: float) -> None:
    all_rows = audit_result.get("all_rows")
    if all_rows is None:
        all_rows = pd.DataFrame()

    def count(col: str) -> int:
        return int(_bool_col(all_rows, col).sum())

    quarterly_mask = _bool_col(all_rows, "quarterly_mismatch_flag")
    quarterly_n = int(quarterly_mask.sum())
    n_critical = 0
    if quarterly_n and "quarterly_severity" in all_rows.columns:
        n_critical = int((all_rows.loc[quarterly_mask, "quarterly_severity"] == "critical").sum())
    quarterly_label = "Несоответствие по кварталам"
    if n_critical:
        quarterly_label += f" (критичных: {n_critical})"

    rows_spec = [
        (count("high_change_flag"),
         f"Резких изменений год-к-году (порог {threshold_pct:.0f}%) не найдено.",
         f"Резкое изменение год-к-году (порог {threshold_pct:.0f}%)"),
        (count("missing_justification_flag"),
         "Все статьи имеют заполненное обоснование.", "Без обоснования"),
        (count("high_price_flag"),
         "Ценовых выбросов не найдено.", "Топ-10% по стоимости (внутри листа)"),
        (quarterly_n,
         "Квартальные суммы согласуются с годовым планом (допуск 1%).", quarterly_label),
        (count("anomalous_code_flag"),
         "Все коды статей соответствуют формату (например IT-CA-01).", "Нестандартный формат кода статьи"),
    ]

    slide = _blank_slide(prs)
    _add_title(slide, "Проблемные и рискованные статьи")

    row_h = Inches(1.0)
    top0 = Inches(1.5)
    for i, (n, zero_message, nonzero_label) in enumerate(rows_spec):
        top = top0 + row_h * i
        indicator, label = ("✓", zero_message) if n == 0 else (str(n), nonzero_label)
        _add_body_text(slide, indicator, Inches(0.8), top, Inches(1.6), row_h,
                        size=36, color=_NAVY, bold=True, align=PP_ALIGN.CENTER)
        _add_body_text(slide, label, Inches(2.6), top + Inches(0.25), _SLIDE_W - Inches(3.4), Inches(0.6),
                        size=16, color=_INK)


# --- slide 5: CAPEX vs OPEX -------------------------------------------------------------
def _add_capex_opex_slide(prs: Presentation, capex_opex_df: pd.DataFrame) -> None:
    if capex_opex_df is None or capex_opex_df.empty:
        return
    plan_label = next((c for c in capex_opex_df.columns if c.startswith("План ")), None)
    if plan_label is None:
        return

    slide = _blank_slide(prs)
    _add_title(slide, "CAPEX vs OPEX")

    chart_data = CategoryChartData()
    chart_data.categories = [str(v) for v in capex_opex_df["Тип"]]
    chart_data.add_series(plan_label, [float(v) if pd.notna(v) else 0.0 for v in capex_opex_df[plan_label]])

    graphic_frame = slide.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(1.2), Inches(1.5), Inches(10.9), Inches(4.6), chart_data
    )
    chart = graphic_frame.chart
    chart.has_legend = False
    plot = chart.plots[0]
    plot.series[0].format.fill.solid()
    plot.series[0].format.fill.fore_color.rgb = _NAVY
    plot.has_data_labels = True
    plot.data_labels.number_format = "#,##0"
    plot.data_labels.number_format_is_linked = False

    # A viewer shouldn't have to eyeball the bars for the real number: state
    # both totals plainly underneath, in the same units as the chart.
    totals_text = " · ".join(
        f'{row["Тип"]}: {_format_amount(row[plan_label])} тыс. тг' for _, row in capex_opex_df.iterrows()
    )
    _add_body_text(slide, totals_text, Inches(1.2), Inches(6.3), Inches(10.9), Inches(0.6),
                    size=16, color=_INK_MUTED, align=PP_ALIGN.CENTER)


# --- slide 6: year trend ---------------------------------------------------------------
def _delta_cell(current, previous) -> str:
    """Table-cell form of analysis.py's _yoy_delta_str() -- reuses its exact
    percentage math, just reformatted from ' (Δ +N.N%)' to '+N.N%' (or '—'
    when there's no prior value of the same kind to compare against)."""
    raw = _yoy_delta_str(current, previous)
    if not raw:
        return "—"
    m = re.search(r"([+-]\d[\d.]*%)", raw)
    return m.group(1) if m else "—"


def _year_trend_deltas(year_trend_df: pd.DataFrame) -> dict[tuple, str]:
    """{(year, kind): delta_cell} vs. the previous year of the SAME kind --
    computable even when every year present is a forecast (no plan/fact
    history before the campaign year): a forecast-to-forecast delta is a
    real, useful number, not a placeholder for a missing baseline."""
    deltas: dict[tuple, str] = {}
    prev_by_kind: dict[str, float] = {}
    for _, row in year_trend_df.sort_values(["Тип", "Год"]).iterrows():
        kind = row["Тип"]
        current = row["Сумма"]
        deltas[(row["Год"], kind)] = _delta_cell(current, prev_by_kind.get(kind))
        prev_by_kind[kind] = current
    return deltas


def _add_year_trend_slide(prs: Presentation, year_trend_df: pd.DataFrame) -> None:
    if year_trend_df is None or year_trend_df.empty:
        return

    wide = year_trend_df.assign(Год=year_trend_df["Год"].astype(str)).pivot_table(
        index="Год", columns="Тип", values="Сумма", aggfunc="sum"
    )
    if wide.empty:
        return

    slide = _blank_slide(prs)
    _add_title(slide, "Динамика по годам")

    chart_data = CategoryChartData()
    chart_data.categories = list(wide.index)
    for col in wide.columns:
        chart_data.add_series(col, [float(v) if pd.notna(v) else 0.0 for v in wide[col]])

    graphic_frame = slide.shapes.add_chart(
        XL_CHART_TYPE.LINE_MARKERS, Inches(0.6), Inches(1.5), Inches(7.4), Inches(5.3), chart_data
    )
    chart = graphic_frame.chart
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.BOTTOM
    chart.legend.include_in_layout = False
    for series in chart.plots[0].series:
        hexcolor = _TREND_COLORS.get(series.name)
        if hexcolor:
            series.format.line.color.rgb = RGBColor.from_string(hexcolor)
            series.format.line.width = Pt(2.5)

    # Compact Год/Тип/Сумма/Δ% table alongside the chart.
    deltas = _year_trend_deltas(year_trend_df)
    ordered = year_trend_df.sort_values(["Год", "Тип"])
    headers = ["Год", "Тип", "Сумма, тыс. тг", "Δ%"]
    n_rows = len(ordered) + 1
    table_shape = slide.shapes.add_table(
        n_rows, len(headers), Inches(8.3), Inches(1.5), Inches(4.4), Inches(0.45) * n_rows
    )
    table = table_shape.table
    for c, header in enumerate(headers):
        _style_table_header(table.cell(0, c), header)
    for r, (_, row) in enumerate(ordered.iterrows(), start=1):
        values = [
            str(row["Год"]), row["Тип"], _format_amount(row["Сумма"]),
            deltas.get((row["Год"], row["Тип"]), "—"),
        ]
        for c, value in enumerate(values):
            _style_table_body_cell(table.cell(r, c), value)


# --- slide 7: supplier concentration (conditional) --------------------------------------
def _add_concentration_slide(prs: Presentation, concentration: dict) -> None:
    top_supplier = (concentration or {}).get("top_supplier") or ""
    top_pct = float((concentration or {}).get("top_supplier_pct") or 0.0)
    flagged = bool((concentration or {}).get("concentration_flag"))
    if not flagged and top_pct < _CONCENTRATION_SLIDE_MIN_PCT:
        return  # no material concentration risk -- a slide here would be noise, not signal

    slide = _blank_slide(prs)
    _add_title(slide, "Концентрация поставщиков")

    _add_body_text(slide, top_supplier or "—", Inches(0.8), Inches(1.6), _SLIDE_W - Inches(1.6),
                    Inches(0.9), size=36, color=_NAVY, bold=True, align=PP_ALIGN.CENTER)
    _add_body_text(slide, f"{top_pct:.1f}% от общего бюджета", Inches(0.8), Inches(2.5),
                    _SLIDE_W - Inches(1.6), Inches(0.6), size=18, color=_INK_MUTED, align=PP_ALIGN.CENTER)

    breakdown = concentration.get("supplier_breakdown")
    if breakdown is None or breakdown.empty or len(breakdown) < 2:
        return
    rest = breakdown.iloc[1:6]  # next 4-5 suppliers by share, after the top one shown above

    headers = ["Поставщик", "Сумма, тыс. тг", "Доля, %"]
    n_rows = len(rest) + 1
    table_shape = slide.shapes.add_table(
        n_rows, len(headers), Inches(1.9), Inches(3.5), Inches(9.5), Inches(0.5) * n_rows
    )
    table = table_shape.table
    for c, header in enumerate(headers):
        _style_table_header(table.cell(0, c), header)
    for r, (_, row) in enumerate(rest.iterrows(), start=1):
        values = [row["counterparty"], _format_amount(row["total_amount"]), f'{row["share_pct"]:.1f}%']
        for c, value in enumerate(values):
            _style_table_body_cell(table.cell(r, c), value)


# --- slide 8+: top risk items (АТК / уточнение, each its own conditional slide) ---------
def _amount_to_float(amount_text: str) -> float:
    """Reverses conclusion_generator._format_amount's space-grouping so rows
    can be ranked by amount; malformed text sorts last rather than raising."""
    try:
        return float(amount_text.replace(" ", "") or 0)
    except ValueError:
        return 0.0


def _add_risk_rows_slide(prs: Presentation, rows: list[dict], title: str) -> None:
    if not rows:
        return
    top = sorted(rows, key=lambda r: _amount_to_float(r["amount"]), reverse=True)[:_MAX_RISK_ROWS]

    slide = _blank_slide(prs)
    _add_title(slide, title)

    headers = ["Лист", "Статья", "Сумма, тыс. тг", "Основание"]
    n_rows = len(top) + 1
    table_shape = slide.shapes.add_table(
        n_rows, len(headers), Inches(0.6), Inches(1.5), _SLIDE_W - Inches(1.2), Inches(0.5) * n_rows
    )
    table = table_shape.table
    for c, header in enumerate(headers):
        _style_table_header(table.cell(0, c), header)
    for r, row in enumerate(top, start=1):
        values = [row["tab"], row["item"], row["amount"], row["reason"]]
        for c, value in enumerate(values):
            _style_table_body_cell(table.cell(r, c), value)


def _default_output_path(audit_result: dict) -> str:
    entity_slug = re.sub(r"[^\w\-]+", "_", str(audit_result.get("entity") or "conclusion")).strip("_")
    year = audit_result.get("budget_year") or ""
    return f"{entity_slug or 'conclusion'}_zakluchenie_{year}.pptx"


def generate_pptx_conclusion(
    audit_result: dict,
    output_path: str | None = None,
    capex_opex_df: pd.DataFrame | None = None,
    year_trend_df: pd.DataFrame | None = None,
    threshold_pct: float | None = None,
) -> str:
    """Render audit_result as a .pptx Заключение; returns the written file's path.

    capex_opex_df/year_trend_df are optional pre-computed tables (as produced
    by core.budget_auditor.parser's capex_opex_table()/year_trend()); when
    omitted they're recomputed from audit_result["all_rows"] if present, and
    the corresponding chart slide is simply left out otherwise. threshold_pct
    is the YoY threshold the caller actually flagged rows at (app.py's
    yoy_threshold), shown on the risk-overview slide; defaults to the same
    15% app.py itself defaults to when not passed.
    """
    if output_path is None:
        output_path = _default_output_path(audit_result)
    output_path = str(Path(output_path).resolve())
    if threshold_pct is None:
        threshold_pct = _DEFAULT_THRESHOLD_PCT

    combined = audit_result.get("all_rows")
    if capex_opex_df is None:
        capex_opex_df = capex_opex_table(combined) if combined is not None else pd.DataFrame()
    if year_trend_df is None:
        year_trend_df = year_trend(combined) if combined is not None else pd.DataFrame()

    narrative = audit_result.get("narrative") or {}
    risk_level = (narrative.get("risk_level") or "").strip()
    if risk_level == "неизвестно":  # fallback placeholder, not a real finding
        risk_level = ""
    executive_summary = (narrative.get("executive_summary") or "").strip()

    prs = Presentation()
    prs.slide_width = _SLIDE_W
    prs.slide_height = _SLIDE_H

    _add_title_slide(prs, audit_result, risk_level)
    _add_executive_summary_slide(prs, audit_result, executive_summary)
    _add_summary_slide(prs, audit_result.get("summary", {}) or {})
    _add_risk_overview_slide(prs, audit_result, threshold_pct)
    _add_capex_opex_slide(prs, capex_opex_df)
    _add_year_trend_slide(prs, year_trend_df)
    _add_concentration_slide(prs, (audit_result.get("findings", {}) or {}).get("concentration", {}) or {})
    _add_risk_rows_slide(prs, _atk_rows(audit_result), "Выносится на АТК")
    _add_risk_rows_slide(prs, _clarification_rows(audit_result), "Требует уточнения")

    prs.save(output_path)
    return output_path
