"""Russian natural-language query engine over the procurement warehouse.

Flow: a Russian question -> LLM writes SQLite SQL (grounded in the warehouse
schema description) -> SQL runs against the warehouse -> LLM phrases the answer
in Russian using ONLY the numbers returned by the database. The model never
invents figures; every number in the answer comes from the query result.

The LLM call itself is routed through legacy.llm_providers.get_provider():
Gemini via its native SDK (`google-genai`) by default, or OpenRouter (openai
SDK) when OPENROUTER_API_KEY is set. See llm_providers.py for the provider
interface and the shared retry/backoff + DailyQuotaExhausted contract.

Per SPEC.md architecture rule: this module is pure — no streamlit imports.
"""

from __future__ import annotations

from legacy import warehouse
from legacy.llm_providers import DailyQuotaExhausted, LLMProvider, get_provider


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
        f"{data_instruction}"
    )

    return _generate_content(client, prompt).strip()


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
        message = "Дневной лимит бесплатного API исчерпан. Попробуйте позже."
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
