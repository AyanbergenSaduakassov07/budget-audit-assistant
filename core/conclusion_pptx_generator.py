"""Render a build_audit_result() dict as a Заключение по результатам
экспертизы .pptx — a slide-deck companion to conclusion_generator's .docx,
built from the exact same audit_result shape (no second data-mapping path).

Row/summary selection is reused from conclusion_generator wherever it
already exists there (_atk_rows, _summary_rows, _format_amount) rather than
re-derived here. CAPEX/OPEX and year-trend chart data come from
core.budget_auditor.parser's own capex_opex_table()/year_trend() — called
either on a caller-supplied table or recomputed from audit_result["all_rows"]
(which carries the same columns as the combined DataFrame those functions
expect; see narrator_bridge._decorate_rows).

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
from pptx.util import Emu, Inches, Pt

from core.budget_auditor.parser import capex_opex_table, year_trend
from core.conclusion_generator import _atk_rows, _format_amount, _summary_rows

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


# --- slide 1: title -----------------------------------------------------------------
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


# --- slide 2: KPI summary (Резюме) --------------------------------------------------
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


# --- slide 3: CAPEX vs OPEX -----------------------------------------------------------
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
        XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(1.2), Inches(1.5), Inches(10.9), Inches(5.3), chart_data
    )
    chart = graphic_frame.chart
    chart.has_legend = False
    plot = chart.plots[0]
    plot.series[0].format.fill.solid()
    plot.series[0].format.fill.fore_color.rgb = _NAVY


# --- slide 4: year trend ---------------------------------------------------------------
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
        XL_CHART_TYPE.LINE_MARKERS, Inches(1.2), Inches(1.5), Inches(10.9), Inches(5.3), chart_data
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


# --- slide 5: top risk (АТК) items -----------------------------------------------------
def _amount_to_float(amount_text: str) -> float:
    """Reverses conclusion_generator._format_amount's space-grouping so rows
    can be ranked by amount; malformed text sorts last rather than raising."""
    try:
        return float(amount_text.replace(" ", "") or 0)
    except ValueError:
        return 0.0


def _add_top_risk_slide(prs: Presentation, atk_rows: list[dict]) -> None:
    if not atk_rows:
        return
    top = sorted(atk_rows, key=lambda r: _amount_to_float(r["amount"]), reverse=True)[:_MAX_RISK_ROWS]

    slide = _blank_slide(prs)
    _add_title(slide, "Наиболее значимые статьи (АТК)")

    headers = ["Лист", "Статья", "Сумма, тыс. тг", "Основание"]
    n_rows = len(top) + 1
    table_shape = slide.shapes.add_table(
        n_rows, len(headers), Inches(0.6), Inches(1.5), _SLIDE_W - Inches(1.2), Inches(0.5) * n_rows
    )
    table = table_shape.table
    for c, header in enumerate(headers):
        cell = table.cell(0, c)
        cell.text = header
        cell.fill.solid()
        cell.fill.fore_color.rgb = _NAVY
        para = cell.text_frame.paragraphs[0]
        para.font.bold = True
        para.font.color.rgb = _WHITE
        para.font.size = Pt(14)

    for r, row in enumerate(top, start=1):
        values = [row["tab"], row["item"], row["amount"], row["reason"]]
        for c, value in enumerate(values):
            cell = table.cell(r, c)
            cell.text = str(value)
            cell.fill.solid()
            cell.fill.fore_color.rgb = _WHITE
            para = cell.text_frame.paragraphs[0]
            para.font.size = Pt(12)
            para.font.color.rgb = _INK


# --- slide 6: closing / executive summary ----------------------------------------------
def _add_closing_slide(prs: Presentation, executive_summary: str) -> None:
    if not executive_summary:
        return
    slide = _blank_slide(prs)
    _add_title(slide, "Итоговое заключение")
    _add_body_text(slide, executive_summary, Inches(0.8), Inches(1.6), _SLIDE_W - Inches(1.6),
                    Inches(5.0), size=18, color=_INK)


def _default_output_path(audit_result: dict) -> str:
    entity_slug = re.sub(r"[^\w\-]+", "_", str(audit_result.get("entity") or "conclusion")).strip("_")
    year = audit_result.get("budget_year") or ""
    return f"{entity_slug or 'conclusion'}_zakluchenie_{year}.pptx"


def generate_pptx_conclusion(
    audit_result: dict,
    output_path: str | None = None,
    capex_opex_df: pd.DataFrame | None = None,
    year_trend_df: pd.DataFrame | None = None,
) -> str:
    """Render audit_result as a .pptx Заключение; returns the written file's path.

    capex_opex_df/year_trend_df are optional pre-computed tables (as produced
    by core.budget_auditor.parser's capex_opex_table()/year_trend()); when
    omitted they're recomputed from audit_result["all_rows"] if present, and
    the corresponding chart slide is simply left out otherwise.
    """
    if output_path is None:
        output_path = _default_output_path(audit_result)
    output_path = str(Path(output_path).resolve())

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
    _add_summary_slide(prs, audit_result.get("summary", {}) or {})
    _add_capex_opex_slide(prs, capex_opex_df)
    _add_year_trend_slide(prs, year_trend_df)
    _add_top_risk_slide(prs, _atk_rows(audit_result))
    _add_closing_slide(prs, executive_summary)

    prs.save(output_path)
    return output_path
