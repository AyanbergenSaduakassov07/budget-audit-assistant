"""SQLite warehouse for the group-wide procurement data.

Loads the flat DataFrame produced by synth.generate_group() into a local SQLite
database so the query engine can run SQL against it, and exposes a read-only
query helper plus a schema description that grounds the language model's SQL.

Per SPEC.md architecture rule: this module is pure — no streamlit imports.
Built-in sqlite3 only, no ORM / SQLAlchemy.
"""

from __future__ import annotations

import sqlite3

import pandas as pd

# Column order of the `procurements` table, matching GROUP_COLUMNS from synth.
_TABLE_COLUMNS = [
    "subsidiary",
    "year",
    "item_name",
    "supplier",
    "category",
    "qty",
    "unit_price",
    "subtotal",
]

_CREATE_TABLE = """
CREATE TABLE procurements (
    subsidiary TEXT,
    year       INTEGER,
    item_name  TEXT,
    supplier   TEXT,
    category   TEXT,
    qty        INTEGER,
    unit_price REAL,
    subtotal   REAL
)
"""

_INSERT = (
    "INSERT INTO procurements "
    "(subsidiary, year, item_name, supplier, category, qty, unit_price, subtotal) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
)


def build_warehouse(df: pd.DataFrame, db_path: str = "procurement.db") -> str:
    """Load `df` into a fresh `procurements` table in the SQLite db at `db_path`.

    Idempotent: drops and recreates the table (and its indexes) every call, so
    repeated builds from the same DataFrame always yield the same warehouse.
    Returns db_path.
    """
    # Cast out of numpy scalar types — sqlite3 only binds native Python types.
    rows = [
        (
            str(r.subsidiary),
            int(r.year),
            str(r.item_name),
            str(r.supplier),
            str(r.category),
            int(r.qty),
            float(r.unit_price),
            float(r.subtotal),
        )
        for r in df[_TABLE_COLUMNS].itertuples(index=False)
    ]

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS procurements")
        cur.execute(_CREATE_TABLE)
        cur.executemany(_INSERT, rows)
        cur.execute("CREATE INDEX idx_procurements_subsidiary ON procurements(subsidiary)")
        cur.execute("CREATE INDEX idx_procurements_year ON procurements(year)")
        cur.execute("CREATE INDEX idx_procurements_category ON procurements(category)")
        conn.commit()
    finally:
        conn.close()

    return db_path


def run_sql(sql: str, db_path: str = "procurement.db") -> pd.DataFrame:
    """Execute a read-only SELECT query and return the result as a DataFrame.

    Hard safety guard: raises ValueError for anything that is not a single
    SELECT statement (blocks INSERT/UPDATE/DELETE/DROP/ALTER, etc.). The query
    also runs against a connection opened in read-only mode as defence in depth.
    """
    if sql is None or not sql.strip():
        raise ValueError("Empty SQL query.")

    # Normalise: strip whitespace and a single trailing semicolon.
    cleaned = sql.strip().rstrip(";").strip()

    # Reject stacked statements like "SELECT ...; DROP TABLE ...".
    if ";" in cleaned:
        raise ValueError("Only a single SELECT statement is allowed (no ';').")

    first_token = cleaned.split(None, 1)[0].upper() if cleaned else ""
    # Allow plain SELECT and read-only WITH ... SELECT common table expressions.
    if first_token not in ("SELECT", "WITH"):
        raise ValueError(
            f"Only read-only SELECT queries are allowed, got: {first_token or '<empty>'}"
        )

    # Open read-only so the engine itself refuses any write that slips through.
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return pd.read_sql_query(cleaned, conn)
    finally:
        conn.close()


def get_schema_description(db_path: str = "procurement.db") -> str:
    """Return a plain-text schema description to ground the LLM's SQL generation.

    Includes every column with its type and meaning, plus the EXACT distinct
    subsidiary names, category values (Russian), and years as actually stored —
    so a Russian-language question maps to the exact string values in the data.
    Accuracy of this text drives the accuracy of the whole query engine.
    """
    subsidiaries = run_sql(
        "SELECT DISTINCT subsidiary FROM procurements ORDER BY subsidiary", db_path
    )["subsidiary"].tolist()
    categories = run_sql(
        "SELECT DISTINCT category FROM procurements ORDER BY category", db_path
    )["category"].tolist()
    years = run_sql(
        "SELECT DISTINCT year FROM procurements ORDER BY year", db_path
    )["year"].tolist()

    subs_block = "\n".join(f"  - {s}" for s in subsidiaries)
    cats_block = "\n".join(f"  - {c}" for c in categories)
    years_str = ", ".join(str(y) for y in years)

    return f"""\
Table: procurements
A flat table of group procurement line items. One row = one purchased line
item for one subsidiary in one year. All monetary values are in national
tenge (KZT).

Columns:
  - subsidiary  TEXT    Name of the ACM subsidiary that made the purchase.
  - year        INTEGER Procurement year (four-digit).
  - item_name   TEXT    Name of the purchased item (lowercase, e.g. 'microsoft 365 e3').
  - supplier    TEXT    Name of the supplier/vendor (lowercase).
  - category    TEXT    Procurement category. VALUES ARE IN RUSSIAN (see below).
  - qty         INTEGER Quantity purchased (integer, > 0).
  - unit_price  REAL    Price per unit in KZT.
  - subtotal    REAL    Line total in KZT (normally qty * unit_price).

Exact distinct subsidiary values (match spelling EXACTLY in WHERE clauses):
{subs_block}

Exact distinct category values — these are Russian strings, use them verbatim:
{cats_block}
    (лицензия = licenses, оборудование = hardware, услуга = services,
     прочее = other/miscellaneous)

Years present: {years_str}

IMPORTANT — matching the user's language to stored values:
When the user asks a question in Russian, the SQL must filter on these EXACT
stored string values, not translations. For example:
  - A question about hardware ("оборудование") -> WHERE category = 'оборудование'
  - A question about licenses ("лицензии") -> WHERE category = 'лицензия'
  - A question about a specific subsidiary -> match the exact stored spelling
    from the list above (e.g. WHERE subsidiary = 'Subsidiary E').
Do not invent category names or subsidiary spellings that are not in the lists
above. Total spend is SUM(subtotal); use GROUP BY for per-subsidiary,
per-year, or per-category breakdowns.
"""


if __name__ == "__main__":
    from core.synth import generate_group

    df, _ = generate_group()
    db_path = build_warehouse(df)
    print(f"Warehouse built at: {db_path} ({len(df)} rows)\n")

    print("=== schema description ===")
    print(get_schema_description(db_path))

    print("=== sample query: total spend per subsidiary, 2024 ===")
    q1 = run_sql(
        "SELECT subsidiary, SUM(subtotal) AS total_spend "
        "FROM procurements WHERE year = 2024 "
        "GROUP BY subsidiary ORDER BY total_spend DESC",
        db_path,
    )
    print(q1.to_string(index=False))

    print("\n=== sample query: hardware ('оборудование') spend per year ===")
    q2 = run_sql(
        "SELECT year, SUM(subtotal) AS hardware_spend "
        "FROM procurements WHERE category = 'оборудование' "
        "GROUP BY year ORDER BY year",
        db_path,
    )
    print(q2.to_string(index=False))
