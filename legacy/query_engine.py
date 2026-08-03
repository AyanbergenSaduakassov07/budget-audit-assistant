"""Russian natural-language query engine over the procurement warehouse.

Flow: a Russian question -> LLM writes SQLite SQL (grounded in the warehouse
schema description) -> SQL runs against the warehouse -> LLM phrases the answer
in Russian using ONLY the numbers returned by the database. The model never
invents figures; every number in the answer comes from the query result.

The LLM call itself is routed through legacy.llm_providers.get_provider():
OpenRouter (openai SDK) when OPENROUTER_API_KEY is set — the only provider
this codebase supports. See llm_providers.py for the provider interface and
the retry/backoff + DailyQuotaExhausted contract.

next_step()/self_check_answer() are schema-agnostic building blocks for a
bounded step loop (propose one SQL query + a reason, or finalize, given
everything queried so far in this same answer) — used by
core.budget_auditor.file_sql.answer_over_file() so the model can react to
what an early query reveals instead of planning every query up front.

Per SPEC.md architecture rule: this module is pure — no streamlit imports.
"""

from __future__ import annotations

import json
import re

from legacy import warehouse
from legacy.llm_providers import (
    DAILY_QUOTA_MESSAGE,
    DailyQuotaExhausted,
    LLMProvider,
    get_provider,
)


def _get_client() -> LLMProvider:
    """Build the active LLM provider, or fail with a clear message."""
    return get_provider()


def _generate_content(client: LLMProvider, prompt: str) -> str:
    """Generate text from a prompt via the active provider."""
    return client.generate_json(prompt)


# Few-shot examples pairing Russian questions with the SQL they should produce.
# They deliberately exercise the Russian category values so the model learns to
# filter on the exact stored strings rather than translating them.
_SQL_EXAMPLES = [
    (
        "Сколько потратили на оборудование в 2024?",
        "SELECT SUM(subtotal) FROM procurements "
        "WHERE category = 'оборудование' AND year = 2024",
    ),
    (
        "Какая общая сумма закупок по всем годам?",
        "SELECT SUM(subtotal) FROM procurements",
    ),
    (
        "Сколько потратил Subsidiary D на лицензии в 2023?",
        "SELECT SUM(subtotal) FROM procurements "
        "WHERE subsidiary = 'Subsidiary D' AND category = 'лицензия' AND year = 2023",
    ),
    (
        "Сравни общие расходы Subsidiary E и Subsidiary I в 2025.",
        "SELECT subsidiary, SUM(subtotal) AS total_spend FROM procurements "
        "WHERE subsidiary IN ('Subsidiary E', 'Subsidiary I') AND year = 2025 "
        "GROUP BY subsidiary",
    ),
]


# Prior conversation turns let a follow-up like "а за все года" be understood
# in context (e.g. "the previous OPEX/CAPEX breakdown, but for every year")
# instead of as a standalone query. Text only -- already-rendered question and
# answer strings from the chat history, never raw data, so this doesn't touch
# the existing data-minimization boundary. Capped here (not just by the
# caller) so any caller passing a longer history still can't inflate prompt
# size/cost; older turns are dropped, not summarized.
_MAX_HISTORY_TURNS = 3


def _format_history(history: list[tuple[str, str]] | None) -> str:
    """Last _MAX_HISTORY_TURNS (question, answer) pairs as a labeled prompt
    block, or '' if there's no history to include."""
    if not history:
        return ""
    recent = history[-_MAX_HISTORY_TURNS:]
    turns = "\n\n".join(f"Вопрос: {q}\nОтвет: {a}" for q, a in recent)
    return (
        "Предыдущие вопросы и ответы в этом диалоге (используй как контекст, "
        "если текущий вопрос ссылается на них, например «а за все года» или "
        "«а по кварталам»):\n\n"
        f"{turns}\n\n---\n\n"
    )


def _strip_sql_fences(text: str) -> str:
    """Remove accidental markdown code fences / labels from an SQL response."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Drop the opening fence line (``` or ```sql) and the closing fence.
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned.strip()


def _generate_sql(
    client: LLMProvider,
    question: str,
    schema_description: str,
    history: list[tuple[str, str]] | None = None,
    examples: list[tuple[str, str]] | None = None,
) -> str:
    """Ask the model to turn a Russian question into a single SQLite SELECT.

    examples: few-shot (question, SQL) pairs shown to the model. Defaults to
    _SQL_EXAMPLES (the synthetic `procurements` warehouse) so the answer()
    call path below is unchanged; callers grounded in a different schema
    (e.g. core.budget_auditor.file_sql's `budget_items` table) must pass
    their own set — otherwise the model sees a correct schema description
    contradicted by worked examples for a table that doesn't exist there.
    """
    if examples is None:
        examples = _SQL_EXAMPLES
    example_block = "\n\n".join(
        f"Вопрос: {q}\nSQL: {sql}" for q, sql in examples
    )

    prompt = (
        "You are a SQL generator for a procurement database (SQLite). "
        "Given a question in Russian, return exactly ONE valid SQLite SELECT "
        "query that answers it. Rules:\n"
        "- Output ONLY the SQL. No explanation, no comments, no markdown fences.\n"
        "- Use only the table and columns from the schema below.\n"
        "- Category and subsidiary values are stored verbatim (categories are in "
        "Russian). Filter on the EXACT stored strings shown in the schema; never "
        "translate or invent them.\n"
        "- Monetary totals use SUM(subtotal).\n\n"
        f"{schema_description}\n"
        "Examples:\n"
        f"{example_block}\n\n"
        f"{_format_history(history)}"
        f"Вопрос: {question}\nSQL:"
    )

    return _strip_sql_fences(client.generate_sql(prompt))


def _generate_sql_with_error_feedback(
    client: LLMProvider,
    question: str,
    schema_description: str,
    failed_sql: str,
    error_message: str,
    history: list[tuple[str, str]] | None = None,
) -> str:
    """One-shot self-correction: tell the model its previous SQL failed with
    this exact error, and ask for a single corrected SQLite SELECT.

    Used by callers that want a bounded retry after warehouse.run_sql()
    raises (see core.budget_auditor.file_sql.answer_over_file) — not part of
    the initial-attempt path, which stays on the plain _generate_sql().
    """
    prompt = (
        "You are a SQL generator for a procurement database (SQLite). "
        "Your previous attempt at writing a SQLite SELECT query for this "
        "question failed when executed. Fix it and return exactly ONE valid "
        "SQLite SELECT query that answers the question. Rules:\n"
        "- Output ONLY the SQL. No explanation, no comments, no markdown fences.\n"
        "- Use only the table and columns from the schema below.\n"
        "- Category and subsidiary/branch values are stored verbatim; filter "
        "on the EXACT stored strings shown in the schema; never translate or "
        "invent them.\n\n"
        f"{schema_description}\n"
        f"{_format_history(history)}"
        f"Вопрос: {question}\n\n"
        f"Предыдущий SQL (не выполнился):\n{failed_sql}\n\n"
        f"Ошибка выполнения:\n{error_message}\n\n"
        "Исправленный SQL:"
    )

    return _strip_sql_fences(client.generate_sql(prompt))


def synthesize_answer(
    client: LLMProvider,
    original_question: str,
    subresults: list[tuple[str, str, str]],
    history: list[tuple[str, str]] | None = None,
) -> str:
    """Combine (sub_question, sql, result_text) triples into one final
    Russian answer to the original compound question.

    Same anti-hallucination discipline as _phrase_answer: the model is told
    to cite ONLY numbers that appear in the sub-results below.
    """
    blocks = "\n\n".join(
        f"Подвопрос: {q}\nSQL: {sql}\nРезультат:\n{result_text}"
        for q, sql, result_text in subresults
    )
    prompt = (
        "Ты аналитик по закупкам компании. Ниже — исходный составной вопрос, "
        "разбитый на подвопросы, и для каждого подвопроса — выполненный SQL и "
        "результат его выполнения. Объедини их в один краткий ответ на "
        "русском на исходный вопрос. Используй ТОЛЬКО числа из результатов "
        "ниже — ничего не выдумывай. Суммы в тенге (KZT). "
        + _NUMBER_FORMAT_INSTRUCTION
        + " " + _FORMATTING_INSTRUCTION
        + " " + _TABLE_FORMAT_INSTRUCTION
        + "\n\n"
        f"{_format_history(history)}"
        f"Исходный вопрос: {original_question}\n\n"
        f"{blocks}\n\n"
        "Итоговый ответ:"
    )
    return _strip_decorative_chars(_generate_content(client, prompt).strip())


# SQL result frames can hold very large sums (e.g. workbook-wide totals in
# raw тенге); pandas' df.to_string() renders those in scientific notation
# (e.g. "1.5e+11"), and a model asked to "use only the numbers from the
# result" can copy that notation straight into the Russian answer. Applies
# to both query_engine.answer() and the file_sql fallback (_phrase_answer is
# shared by both — see core.budget_auditor.file_sql).
_NUMBER_FORMAT_INSTRUCTION = (
    "Число пиши полностью, с разделителями разрядов (пробел между группами "
    "по три цифры, например 155 903 400 000), а НЕ в экспоненциальной записи "
    "(никогда не пиши вида 1.5e+11)."
)

# Shared across every LLM-generated user-facing Russian text in this codebase
# (this module's _phrase_answer/synthesize_answer, and
# core.budget_narrator._build_prompt, which imports this constant): a
# generated business/audit document must read like one, not like chat output.
# Stated as an explicit line on every prompt rather than assumed from context —
# a model that isn't told not to decorate its answer sometimes does.
_FORMATTING_INSTRUCTION = (
    "Пиши официальным деловым языком, без эмодзи и любых декоративных "
    "символов, без markdown-разметки (никаких **, *, -, # для форматирования) "
    "— обычным связным текстом, как в официальном документе."
)

# Single-value and short-list answers should stay prose (_FORMATTING_INSTRUCTION
# above), but a multi-year or multi-item comparison crammed into one paragraph
# is unreadable ("2025 (Факт): OPEX = ..., CAPEX = ...; 2026 (План): OPEX = ...
# " repeated seven times run-on). Used only by synthesize_answer/next_step
# (compound, multi-row answers) -- not _phrase_answer (usually a single
# number) or the narrator's executive_summary (meant to stay short prose).
_TABLE_FORMAT_INSTRUCTION = (
    "Единственное исключение из запрета на markdown выше: если ответ "
    "сравнивает 3 и более однотипных строк (несколько лет, несколько статей, "
    "план/факт/прогноз по годам и т.п.), оформи это как markdown-таблицу — "
    "строка заголовков, строка-разделитель (|---|---|), затем строки данных, "
    "с отдельной колонкой на Δ % там, где применимо. Если сравниваются меньше "
    "3 строк или ответ — одно число, пиши обычным предложением без таблицы."
)

# Second line of defense: strip decorative characters a model produced anyway,
# despite _FORMATTING_INSTRUCTION. Covers the two emoji blocks most commonly
# seen in LLM output (misc symbols/pictographs, misc symbols & dingbats) plus
# the invisible modifiers that ride along with them (variation selector-16,
# zero-width joiner) so removing an emoji doesn't leave an orphaned modifier
# behind.
_EMOJI_RE = re.compile(
    "[" "\U0001f300-\U0001faff" "\U00002600-\U000027bf" "\U0000fe0f" "\U0000200d" "]",
    flags=re.UNICODE,
)
_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_MD_ITALIC_STAR_RE = re.compile(r"(?<!\w)\*(.+?)\*(?!\w)")
_MD_ITALIC_UNDERSCORE_RE = re.compile(r"(?<!\w)_(.+?)_(?!\w)")
_MD_BULLET_RE = re.compile(r"^[ \t]*[-*][ \t]+")

# Third line of defense (alongside _NUMBER_FORMAT_INSTRUCTION): a model can
# still copy a raw pandas/Python float straight into its answer despite being
# told to format numbers with space-grouped thousands -- e.g. "59806.066590896546"
# or, with a Russian decimal comma, "5,001622%". This codebase's own numeric
# convention never uses "." or "," for thousands-grouping (always a space, per
# _NUMBER_FORMAT_INSTRUCTION's example), so any digits-separator-digits run
# with 3+ decimal digits is unambiguously leaked float precision, never a
# legitimate grouped integer. Weaker/differently-tuned models (observed
# switching from Qwen3.6 to a free substitute) follow the prompt instruction
# less reliably, so this is a deterministic backstop, not a replacement for it.
_RAW_DECIMAL_RE = re.compile(r"\d+([.,])\d{3,}")


def _round_raw_decimals(text: str) -> str:
    def _round(m: re.Match) -> str:
        sep = m.group(1)
        value = float(m.group().replace(",", "."))
        rounded = f"{value:.1f}".rstrip("0").rstrip(".")
        return rounded.replace(".", sep) if sep == "," else rounded

    return _RAW_DECIMAL_RE.sub(_round, text)


def _strip_decorative_chars(text: str) -> str:
    """Plain running text: emoji, markdown bold/italic/bullet markers, and
    leaked raw-float precision removed. A model that ignores
    _FORMATTING_INSTRUCTION or _NUMBER_FORMAT_INSTRUCTION still can't leak
    decoration or unrounded numbers into a document meant to read as an
    official conclusion."""
    if not text:
        return text
    cleaned = _EMOJI_RE.sub("", text)
    cleaned = _MD_BOLD_RE.sub(r"\1", cleaned)
    cleaned = _MD_ITALIC_STAR_RE.sub(r"\1", cleaned)
    cleaned = _MD_ITALIC_UNDERSCORE_RE.sub(r"\1", cleaned)
    cleaned = "\n".join(_MD_BULLET_RE.sub("", line) for line in cleaned.splitlines())
    cleaned = _round_raw_decimals(cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


# --- bounded step loop (propose-a-query-or-finalize, given prior steps) -----------
def _format_scratchpad(scratchpad: list[dict]) -> str:
    """Prior {sql, result, reason} steps from THIS SAME answer, as a labeled
    prompt block -- '' when nothing has been queried yet (step 1), so the
    prompt doesn't carry an empty "Уже выполненные шаги:" header."""
    if not scratchpad:
        return ""
    blocks = "\n\n".join(
        f"Шаг {i}: {entry.get('reason') or '(причина не указана)'}\n"
        f"SQL: {entry['sql']}\nРезультат:\n{entry['result']}"
        for i, entry in enumerate(scratchpad, start=1)
    )
    return f"Уже выполненные шаги в этом ответе:\n\n{blocks}\n\n"


def _parse_step_response(raw_text: str) -> dict | None:
    """Parse a next_step()/self_check_answer() JSON response; None (never
    raise) on anything malformed, so the caller can stop looping / fall back
    rather than crash on garbage. _strip_sql_fences is a generic ``` fence
    stripper despite its name -- reused here rather than a second copy."""
    cleaned = _strip_sql_fences(raw_text)
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _build_step_prompt(
    question: str,
    schema_description: str,
    scratchpad: list[dict],
    history: list[tuple[str, str]] | None,
    examples: list[tuple[str, str]] | None,
) -> str:
    if examples is None:
        examples = _SQL_EXAMPLES
    example_block = "\n\n".join(f"Вопрос: {q}\nSQL: {sql}" for q, sql in examples)
    must_query_note = (
        "Выше нет ни одного выполненного шага — тебе ОБЯЗАТЕЛЬНО нужно "
        "сначала выполнить хотя бы один SQL-запрос; финальный ответ без "
        "единого запроса недопустим.\n\n"
        if not scratchpad else ""
    )
    return (
        "Ты аналитик по закупкам компании. Отвечаешь на вопрос пользователя, "
        "выполняя SQL-запросы к базе данных (SQLite) по одному за раз. На "
        "каждом шаге у тебя есть результаты всех предыдущих запросов этого "
        "же ответа — используй их: если для следующего запроса нужно "
        "значение из предыдущего результата (например, сначала найти нужное "
        "подразделение, затем проверить что-то по нему), подставь это "
        "значение напрямую, а не выдумывай его.\n\n"
        f"{schema_description}\n"
        "Примеры вопросов и SQL:\n"
        f"{example_block}\n\n"
        f"{_format_history(history)}"
        f"Исходный вопрос: {question}\n\n"
        f"{_format_scratchpad(scratchpad)}"
        f"{must_query_note}"
        "Реши: нужен ли ещё один SQL-запрос, или данных выше уже достаточно "
        "для ПОЛНОГО ответа на исходный вопрос (проверь, что покрыты ВСЕ его "
        "части).\n\n"
        "Если нужен ещё запрос — верни его вместе с краткой причиной. Если "
        "данных уже достаточно — дай итоговый ответ на русском, кратко, "
        "указав конкретные числа. Используй ТОЛЬКО числа из результатов "
        "запросов выше — ничего не выдумывай. Суммы в тенге (KZT). "
        + _NUMBER_FORMAT_INSTRUCTION + " " + _FORMATTING_INSTRUCTION
        + " " + _TABLE_FORMAT_INSTRUCTION + "\n\n"
        "Верни ТОЛЬКО валидный JSON без markdown-ограждений (без ```), без "
        "пояснений до или после, ОДНОЙ из двух структур:\n"
        '{"action": "query", "sql": "<один SQLite SELECT>", '
        '"reason": "<кратко, зачем этот запрос>"}\n'
        '{"action": "final", "answer": "<итоговый ответ на русском>"}'
    )


def next_step(
    client: LLMProvider,
    question: str,
    schema_description: str,
    scratchpad: list[dict],
    history: list[tuple[str, str]] | None = None,
    examples: list[tuple[str, str]] | None = None,
) -> dict:
    """One step of the bounded step loop used by
    core.budget_auditor.file_sql.answer_over_file(): given everything
    queried so far in THIS answer (scratchpad, a list of {sql, result,
    reason} dicts), ask the model to either propose one more SQL query (with
    a short reason) or give a final Russian answer.

    Returns one of:
      {"action": "query", "sql": str, "reason": str}
      {"action": "final", "answer": str}
      {"action": "error"}   -- malformed/empty response; caller stops
                               looping and falls back to forced synthesis.
    Never raises.
    """
    prompt = _build_step_prompt(question, schema_description, scratchpad, history, examples)
    parsed = _parse_step_response(_generate_content(client, prompt))
    if parsed is None:
        return {"action": "error"}

    action = parsed.get("action")
    if action == "final":
        answer_text = parsed.get("answer")
        if not isinstance(answer_text, str) or not answer_text.strip():
            return {"action": "error"}
        return {"action": "final", "answer": _strip_decorative_chars(answer_text.strip())}
    if action == "query":
        sql = parsed.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            return {"action": "error"}
        reason = parsed.get("reason")
        return {
            "action": "query",
            "sql": _strip_sql_fences(sql.strip()),
            "reason": reason.strip() if isinstance(reason, str) else "",
        }
    return {"action": "error"}


def self_check_answer(client: LLMProvider, question: str, draft_answer: str) -> dict:
    """Cheap completeness check on a draft answer before it's returned: does
    it address every part of the original question (which may have several
    parts)? A malformed/unparseable response defaults to
    {"complete": True, "missing": ""} -- a self-check failure means "ship
    the draft as-is", never a reason to block the answer or raise."""
    prompt = (
        "Ты проверяешь черновой ответ на вопрос пользователя по базе "
        "закупок компании. Оцени: полностью ли черновой ответ покрывает ВСЕ "
        "части исходного вопроса (вопрос мог состоять из нескольких частей, "
        "а ответ покрывать только одну), или чего-то не хватает.\n\n"
        f"Исходный вопрос: {question}\n\n"
        f"Черновой ответ: {draft_answer}\n\n"
        "Верни ТОЛЬКО валидный JSON без markdown-ограждений, в точности "
        "такой структуры:\n"
        '{"complete": true или false, "missing": "<чего именно не хватает, '
        'если complete=false, иначе пустая строка>"}'
    )
    parsed = _parse_step_response(_generate_content(client, prompt))
    if not isinstance(parsed, dict) or "complete" not in parsed:
        return {"complete": True, "missing": ""}
    missing = parsed.get("missing")
    return {
        "complete": bool(parsed.get("complete")),
        "missing": missing.strip() if isinstance(missing, str) else "",
    }


def _phrase_answer(
    client: LLMProvider,
    question: str,
    sql: str,
    result_text: str,
    empty: bool,
    history: list[tuple[str, str]] | None = None,
) -> str:
    """Ask the model to phrase the DB result as a concise Russian answer."""
    if empty:
        data_instruction = (
            "Результат запроса пуст (данные не найдены). Ответь по-русски, что "
            "данные по этому запросу не найдены. НЕ придумывай никаких чисел."
        )
    else:
        data_instruction = (
            "Ответь на вопрос по-русски, кратко, указав конкретные числа из "
            "результата запроса. Используй ТОЛЬКО числа из результата — ничего "
            "не выдумывай. Суммы в тенге (KZT). " + _NUMBER_FORMAT_INSTRUCTION
        )

    prompt = (
        "Ты аналитик по закупкам компании. Отвечай на русском языке, кратко и по делу. "
        "Все числа бери строго из результата SQL-запроса, ничего не придумывай.\n\n"
        f"{_format_history(history)}"
        f"Вопрос пользователя: {question}\n\n"
        f"Выполненный SQL:\n{sql}\n\n"
        f"Результат запроса:\n{result_text}\n\n"
        f"{data_instruction} {_FORMATTING_INSTRUCTION}"
    )

    return _strip_decorative_chars(_generate_content(client, prompt).strip())


def answer(question: str, db_path: str = "procurement.db") -> dict:
    """Answer a Russian procurement question end to end.

    Returns a dict:
      {"question", "sql", "data" (DataFrame or None), "answer" (Russian text),
       "error" (None or a Russian error message)}.

    The generated SQL is always returned for transparency. Any SQL execution
    failure is caught and surfaced as a Russian error message rather than
    crashing.
    """
    result = {"question": question, "sql": None, "data": None, "answer": None, "error": None}

    client = _get_client()
    schema_description = warehouse.get_schema_description(db_path)

    try:
        # Step 1 — generate SQL.
        sql = _generate_sql(client, question, schema_description)
        result["sql"] = sql

        # Step 2 — execute (never crash on a bad query).
        try:
            df = warehouse.run_sql(sql, db_path)
        except Exception as exc:  # SQL/DB errors surfaced, not raised
            message = f"Не удалось выполнить запрос к данным: {exc}"
            result["error"] = message
            result["answer"] = message
            return result

        result["data"] = df

        # Step 3 — phrase the answer in Russian from the actual result.
        result_text = df.to_string(index=False) if not df.empty else "(пустой результат)"
        result["answer"] = _phrase_answer(client, question, sql, result_text, empty=df.empty)
        return result
    except DailyQuotaExhausted:
        message = DAILY_QUOTA_MESSAGE
        result["error"] = message
        result["answer"] = message
        return result


if __name__ == "__main__":
    from legacy.synth import generate_group

    df, _ = generate_group()
    db_path = warehouse.build_warehouse(df)
    print(f"Warehouse built at: {db_path} ({len(df)} rows)\n")

    questions = [
        "Какая общая сумма всех закупок за все годы?",
        "Сравни общие расходы Subsidiary D и Subsidiary F в 2024 году.",
        "Сколько потратили на оборудование в 2024 году?",
    ]

    for q in questions:
        res = answer(q, db_path)
        print("=" * 70)
        print(f"Вопрос:  {res['question']}")
        print(f"SQL:     {res['sql']}")
        if res["error"]:
            print(f"Ошибка:  {res['error']}")
        print(f"Ответ:   {res['answer']}")
        print()
