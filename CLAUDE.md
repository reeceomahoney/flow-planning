# flow-planning

## Conventions

- **Keep comments short.** Avoid long or multi-line comments; add one only where
  the code is genuinely unclear, and explain *why*, not *what*. Prefer clear
  code and names over commentary. Same for docstrings — concise.
- **No leading-underscore function or method names.** Name everything publicly
  (e.g. `setup_ik`, not `_setup_ik`); dunders like `__init__` are fine.
- Lint/format with `uv run pre-commit run --files <files>` (ruff + ty).
- Tests: `uv run pytest`.
