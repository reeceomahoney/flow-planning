# flow-planning

## Conventions

- **Keep comments short.** Avoid long or multi-line comments; add one only where
  the code is genuinely unclear, and explain *why*, not *what*. Prefer clear
  code and names over commentary. Same for docstrings — concise.
- Lint/format with `uv run pre-commit run --files <files>` (ruff + ty).
- Tests: `uv run pytest`.
