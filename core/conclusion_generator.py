"""Render a run_full_audit() result as a Заключение по результатам экспертизы.

Structure mirrors a real expert-review conclusion: header, three
decision sections (АТК / уточнение / согласовано), and a summary table.
The output is a standard, fully editable .docx — no locked sections, no
content controls — since a employee must review and hand-edit it
before it goes up for signature.

All data wrangling (filtering by decision, formatting numbers, composing
the fixed Russian section text) happens here in Python; scripts/render_conclusion.js
only lays out a pre-built JSON payload with the docx package. That keeps the
one Node dependency this project has to a thin rendering layer.

Row-level "Основание"/"Суть уточнения" text is read from
audit_result["all_rows"]["reason"] — budget_audit.run_full_audit() already
knows which rule claimed each row when it builds atk_idx/utochnenie_idx, so
that's where the wording is decided (once, in one place) rather than
re-derived here per row-source.

One data gap worth flagging: a row can be flagged as уточнение *solely* via
a critical quarterly mismatch, but that source wasn't listed among section
2's population rules (growth_flags/missing_fields/new_items only). Omitting
it would leave the printed уточнение table short of the authoritative
summary["clarification_count"] it's captioned with, so quarterly-mismatch
rows are included here too, whenever run_full_audit's own priority
resolution (all_rows["decision"]) confirms they landed on уточнение rather
than getting absorbed into АТК.

Per SPEC.md architecture rule: this module is pure — no streamlit imports.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

_RENDER_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "render_conclusion.js"

_ATK_DECISION = "АТК"
_CLARIFICATION_DECISION = "уточнение"
_APPROVED_DECISION = "согласовано"


def _clean_text(value) -> str:
    """Blank-safe string form: None/NaN/'nan' all collapse to ''."""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _format_amount(value) -> str:
    """Space-grouped thousands (Russian convention), e.g. '467 936'."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        value = 0
    return f"{float(value):,.0f}".replace(",", " ")


def _item_label(row) -> str:
    """'{article_code} — {name}', degrading gracefully if either is blank.

    name prefers counterparty, but falls back to item_name then statya_name
    when counterparty is blank -- otherwise every counterparty-less row
    sharing the same article_code renders as the same indistinguishable
    "{code} — —" label, even when they're genuinely different line items."""
    article = _clean_text(row.get("article_code"))
    name = (
        _clean_text(row.get("counterparty"))
        or _clean_text(row.get("item_name"))
        or _clean_text(row.get("statya_name"))
        or "—"
    )
    return f"{article} — {name}" if article else name


def _final_decision(all_rows: pd.DataFrame | None, idx) -> str | None:
    """The row's ultimate decision per run_full_audit's own priority resolution."""
    if all_rows is None or "decision" not in all_rows.columns or idx not in all_rows.index:
        return None
    return all_rows.loc[idx, "decision"]


def _reason_for(all_rows: pd.DataFrame | None, idx) -> str:
    """The reason run_full_audit recorded for this row (see module docstring)."""
    if all_rows is None or "reason" not in all_rows.columns or idx not in all_rows.index:
        return ""
    return _clean_text(all_rows.loc[idx, "reason"])


# --- section 1: АТК --------------------------------------------------------------
def _atk_rows(audit_result: dict) -> list[dict]:
    findings = audit_result.get("findings", {}) or {}
    all_rows = audit_result.get("all_rows")
    sheet_name = audit_result.get("sheet_name", "")
    rows: list[dict] = []

    growth = findings.get("growth_flags")
    if growth is not None and not growth.empty:
        for idx, r in growth[growth["decision"] == _ATK_DECISION].iterrows():
            rows.append({
                "tab": sheet_name,
                "item": _item_label(r),
                "amount": _format_amount(r.get("plan_annual")),
                "reason": _reason_for(all_rows, idx),
            })

    concentration = findings.get("concentration", {}) or {}
    if concentration.get("concentration_flag"):
        top_supplier = concentration.get("top_supplier", "")
        top_pct = concentration.get("top_supplier_pct", 0.0)
        amount = 0
        breakdown = concentration.get("supplier_breakdown")
        if breakdown is not None and not breakdown.empty:
            match = breakdown[breakdown["counterparty"] == top_supplier]
            if not match.empty:
                amount = match.iloc[0]["total_amount"]
        rows.append({
            "tab": "сквозной вопрос",
            "item": top_supplier or "—",
            "amount": _format_amount(amount),
            "reason": f"концентрация поставщика: {top_pct:.0f}% от общего бюджета",
        })

    return rows


# --- section 2: уточнение ---------------------------------------------------------
def _clarification_rows(audit_result: dict) -> list[dict]:
    findings = audit_result.get("findings", {}) or {}
    all_rows = audit_result.get("all_rows")
    sheet_name = audit_result.get("sheet_name", "")
    rows: list[dict] = []

    growth = findings.get("growth_flags")
    if growth is not None and not growth.empty:
        for idx, r in growth[growth["decision"] == _CLARIFICATION_DECISION].iterrows():
            rows.append({
                "tab": sheet_name,
                "item": _item_label(r),
                "amount": _format_amount(r.get("plan_annual")),
                "reason": _reason_for(all_rows, idx),
            })

    missing = findings.get("missing_fields")
    if missing is not None and not missing.empty:
        for idx, r in missing.iterrows():
            if _final_decision(all_rows, idx) != _CLARIFICATION_DECISION:
                continue  # e.g. absorbed into АТК by a higher-priority growth flag
            rows.append({
                "tab": sheet_name,
                "item": _item_label(r),
                "amount": _format_amount(r.get("plan_annual")),
                "reason": _reason_for(all_rows, idx),
            })

    new_items = findings.get("new_items")
    if new_items is not None and not new_items.empty:
        for idx, r in new_items.iterrows():
            if _final_decision(all_rows, idx) != _CLARIFICATION_DECISION:
                continue
            rows.append({
                "tab": sheet_name,
                "item": _item_label(r),
                "amount": _format_amount(r.get("plan_annual")),
                "reason": _reason_for(all_rows, idx),
            })

    # See module docstring: without this, a row flagged solely by a critical
    # quarterly mismatch would count toward the footer but never appear in
    # the table above it.
    quarterly = findings.get("quarterly_mismatches")
    if quarterly is not None and not quarterly.empty:
        critical = quarterly[quarterly["severity"] == "critical"]
        for idx, r in critical.iterrows():
            if _final_decision(all_rows, idx) != _CLARIFICATION_DECISION:
                continue
            rows.append({
                "tab": sheet_name,
                "item": _item_label(r),
                "amount": _format_amount(r.get("plan_annual")),
                "reason": _reason_for(all_rows, idx),
            })

    return rows


# --- section 3: согласовано --------------------------------------------------------
def _approved_groups(audit_result: dict) -> list[dict]:
    all_rows = audit_result.get("all_rows")
    sheet_name = audit_result.get("sheet_name", "")
    if all_rows is None or all_rows.empty or "decision" not in all_rows.columns:
        return []

    approved = all_rows[all_rows["decision"] == _APPROVED_DECISION]
    if approved.empty:
        return []

    has_sheet_col = "sheet_name" in approved.columns
    groups: dict[str, list[str]] = {}
    for _, r in approved.iterrows():
        label = (_clean_text(r["sheet_name"]) if has_sheet_col else "") or sheet_name or "—"
        name = _clean_text(r.get("counterparty")) or _clean_text(r.get("article_code")) or "—"
        bucket = groups.setdefault(label, [])
        if name not in bucket:
            bucket.append(name)

    return [{"label": label, "items": items} for label, items in groups.items()]


# --- Резюме -------------------------------------------------------------------------
def _summary_rows(summary: dict) -> list[dict]:
    return [
        {
            "decision": "Выносится на АТК",
            "count": summary.get("atk_count", 0),
            "amount": _format_amount(summary.get("atk_amount", 0)),
            "bold": False,
        },
        {
            "decision": "Требует уточнения",
            "count": summary.get("clarification_count", 0),
            "amount": _format_amount(summary.get("clarification_amount", 0)),
            "bold": False,
        },
        {
            "decision": "Согласовано",
            "count": summary.get("approved_count", 0),
            "amount": _format_amount(summary.get("approved_amount", 0)),
            "bold": False,
        },
        {
            "decision": "ИТОГО",
            "count": summary.get("total_rows", 0),
            "amount": _format_amount(summary.get("total_amount", 0)),
            "bold": True,
        },
    ]


def _build_payload(audit_result: dict) -> dict:
    entity = audit_result.get("entity", "")
    budget_year = audit_result.get("budget_year")
    summary = audit_result.get("summary", {}) or {}
    narrative = audit_result.get("narrative") or {}

    intro_text = (
        f"По результатам рассмотрения статей бюджета ИТ-расходов на {budget_year} год, "
        f"представленных {entity} с обоснованиями, по каждой статье принято одно из трёх "
        "решений: согласовано; требует уточнения, выносится на рассмотрение АТК."
    )

    return {
        "entity": entity,
        "budget_year": budget_year,
        "intro_text": intro_text,
        "executive_summary": narrative.get("executive_summary") or None,
        "atk_section": {
            "intro": (
                "Ввиду существенности сумм и/или системного характера вопросов, "
                "требующих коллегиального решения:"
            ),
            "rows": _atk_rows(audit_result),
            "footer": f"Итого на АТК: {summary.get('atk_count', 0)} статей",
        },
        "clarification_section": {
            "rows": _clarification_rows(audit_result),
            "footer": f"Итого требуют уточнения: {summary.get('clarification_count', 0)} статей",
        },
        "approved_section": {
            "intro": (
                "Рост по данным статьям умеренный либо количественно обоснован, "
                "дополнительных вопросов не требуется:"
            ),
            "groups": _approved_groups(audit_result),
            "footer": f"Итого согласовано: {summary.get('approved_count', 0)} статей",
        },
        "summary_table": {"rows": _summary_rows(summary)},
    }


def _default_output_path(audit_result: dict) -> str:
    entity_slug = re.sub(r"[^\w\-]+", "_", str(audit_result.get("entity") or "conclusion")).strip("_")
    year = audit_result.get("budget_year") or ""
    return f"{entity_slug or 'conclusion'}_zakluchenie_{year}.docx"


def generate_conclusion(audit_result: dict, output_path: str | None = None) -> str:
    """Render audit_result as a .docx Заключение; returns the written file's path.

    Delegates layout to scripts/render_conclusion.js (Node + the docx package)
    over a JSON temp file — this function does all the data prep and never
    hands the Node script anything but plain strings/numbers.
    """
    if output_path is None:
        output_path = _default_output_path(audit_result)
    output_path = str(Path(output_path).resolve())

    payload = _build_payload(audit_result)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(payload, tmp, ensure_ascii=False)
        payload_path = tmp.name

    try:
        result = subprocess.run(
            ["node", str(_RENDER_SCRIPT), payload_path, output_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"docx generation failed: {result.stderr.strip()}")
    finally:
        os.unlink(payload_path)

    return output_path
