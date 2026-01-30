# Patterns & Conventions

Recurring patterns and conventions discovered during development.

## Code Patterns

- **Tool structure:** Each tool in `tools/{name}/main.py` with PEP 723 inline metadata
- **Data flow:** `data/raw/` -> `data/processed/` -> `outputs/`
- **Scratch first:** All exploratory code starts in `scratch/{agent-name}/`

## Naming Conventions

- Tool directories: lowercase with hyphens (`momentum-analyzer`)
- Output files: include timestamp (`{name}_{YYYYMMDD_HHMMSS}.png`)
- Data files: descriptive with date/version if applicable

## Common Pitfalls

(Document recurring issues and their solutions here)
