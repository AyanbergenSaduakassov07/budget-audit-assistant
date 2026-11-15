# Contributing

    uv sync && npm ci
    uv run pytest -m "not api"      # live-API tests are marked `api` and need a provider key

Keep changes small and covered by a test. Tests build their Excel fixtures in code; do not commit real workbooks (`*.xlsx` is gitignored).
