# Work Organization

## Directory Structure

`/tools/` - Each tool gets its own subfolder with main.py entry point
`/shared/` - Shared utilities across tools
`/data/` - Datasets (see datasets.md for details)
`/outputs/` - Generated outputs, reports, charts
`/docs/` - Documentation (see documentation.md for subdirectories)
`/scripts/` - Standalone reusable scripts
`/tests/` - Test suite
`/scratch/` - Temporary exploration (gitignored): `/scratch/{agent-name}/`

Root-level files:
`TASKS.md` - Task tracking (Now/Next/Later/Completed)
`README.md` - Project overview

## File Placement Decision Tree

Input data? -> `/data/` (raw/intermediate/processed)
Tool output? -> `/outputs/{tool-name}/`
Documentation? -> `/docs/` (appropriate subdirectory)
Human exploratory work? -> `/scratch/{your-name}/`
AI exploration/scripts? -> `/scratch/{agent-name}/`
Reusable code? -> `/shared/` (if used across tools) or `/scripts/` (standalone)
New tool? -> `/tools/{tool-name}/`

When uncertain, ASK before creating files.

## Rule: Scratch First

ALL agent-generated files go in `scratch/{agent-name}/{artifact-name}/` first.

**Agent names**: Use the tool name - `claude`, `codex`, `cursor`, `gemini`, `copilot`, etc.

Examples:
- `scratch/claude/fiz-scraper/script.py`
- NOT `scripts/script.py`
- NOT `docs/notes.md`
- NOT `scratch/my-work/file.py` (missing agent-name)

Only create outside scratch if:
1. User explicitly specifies path: "Create X at Y"
2. Editing existing files (not new content)
3. User explicitly requests cleanup/organization of existing code

**After creation**: Notify user where work is located for review.

## When to Promote Files

- Data: Output becomes input to other analyses -> `/data/processed/{dataset-name}/`
- Code: Scratch becomes reusable -> `/shared/` (if used across tools) or `/scripts/` (standalone)
- Output: Report ready to share -> `/outputs/`
- Reference: Output becomes reference material -> `/docs/reference/`

## Common Mistakes

- Scripts in root, mixing raw/processed data, AI files outside scratch, deep nesting, referencing `scratch/` from permanent docs
