"""Russian natural-language query engine over the procurement warehouse.

Flow: a Russian question -> LLM writes SQLite SQL (grounded in the warehouse
schema description) -> SQL runs against the warehouse -> LLM phrases the answer
in Russian using ONLY the numbers returned by the database. The model never
invents figures; every number in the answer comes from the query result.

Uses Google Gemini through its native SDK (`google-genai`), not a vendor-specific SDK.

Per SPEC.md architecture rule: this module is pure — no streamlit imports.
"""

from __future__ import annotations

import os
import time

from google import genai
from google.genai import errors as genai_errors

from core import warehouse

_MODEL = "gemini-flash-lite-latest"

# Retry transient server-side failures (e.g. 503 "high demand", 429 per-minute
# rate limit) with exponential backoff. These are load spikes on Google's side,
# not bugs in the request, so a short wait and retry usually clears them.
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
_MAX_ATTEMPTS = 5
_BASE_DELAY_S = 2.0

# Markers in a 429 error message that identify daily (per-day) quota exhaustion
# rather than a transient per-minute rate limit. A daily limit will not clear on
# a short backoff, so retrying is pointless — we surface it immediately instead.
_DAILY_QUOTA_MARKERS = ("PerDay", "per_day", "RPD")


class DailyQuotaExhausted(Exception):
    """Raised when the free-tier daily API quota is exhausted (not retryable)."""


def _is_daily_quota_error(exc: genai_errors.APIError) -> bool:
    """True if a 429 is a per-day quota exhaustion (vs. a per-minute rate limit)."""
    if exc.code != 429:
        return False
    message = str(exc)
    return any(marker in message for marker in _DAILY_QUOTA_MARKERS)


def _generate_content(client: genai.Client, prompt: str) -> str:
    """Call Gemini with retry + exponential backoff on transient server errors.

    A 429 caused by daily quota exhaustion is not transient — it raises
    DailyQuotaExhausted immediately so the caller can surface a clear message.
    """
    for attempt in range(_MAX_ATTEMPTS):
        try:
            response = client.models.generate_content(model=_MODEL, contents=prompt)
            return response.text or ""
        except genai_errors.APIError as exc:
            if _is_daily_quota_error(exc):
                raise DailyQuotaExhausted(str(exc)) from exc
            transient = exc.code in _RETRY_STATUS
            if not transient or attempt == _MAX_ATTEMPTS - 1:
                raise
            time.sleep(_BASE_DELAY_S * (2**attempt))
    # Unreachable: the loop either returns or raises on the final attempt.
    raise RuntimeError("retry loop exited without a result")

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


def _get_client() -> genai.Client:
    """Build the Gemini client, or fail with a clear message."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Set it to your Google Gemini API key, e.g.:\n"
            "    export GEMINI_API_KEY=your_key_here"
        )
    return genai.Client(api_key=api_key)


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


def _generate_sql(client: genai.Client, question: str, schema_description: str) -> str:
    """Ask the model to turn a Russian question into a single SQLite SELECT."""
    example_block = "\n\n".join(
        f"Вопрос: {q}\nSQL: {sql}" for q, sql in _SQL_EXAMPLES
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
        f"Вопрос: {question}\nSQL:"
    )

    return _strip_sql_fences(_generate_content(client, prompt))


def _phrase_answer(client: genai.Client, question: str, sql: str, result_text: str, empty: bool) -> str:
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
            "не выдумывай. Суммы в тенге (KZT)."
        )

    prompt = (
        "Ты аналитик по закупкам компании. Отвечай на русском языке, кратко и по делу. "
        "Все числа бери строго из результата SQL-запроса, ничего не придумывай.\n\n"
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
    from core.synth import generate_group

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
