---
name: share-gist
description: Share tool code and/or outputs via GitHub Gist
---

# Share via Gist Skill

Creates a GitHub Gist with tool code, outputs, or both.

## Prerequisites

GitHub CLI must be authenticated. In Claude Code Cloud, this happens automatically via the GitHub proxy.

## Usage Patterns

### Share just the tool code

```bash
gh gist create tools/{tool-name}/main.py \
    --public \
    -d "{tool-name}: {brief description}"
```

### Share tool + README

```bash
gh gist create \
    tools/{tool-name}/main.py \
    tools/{tool-name}/README.md \
    --public \
    -d "{tool-name}: {brief description}"
```

### Share tool + output

```bash
gh gist create \
    tools/{tool-name}/main.py \
    outputs/{output-file} \
    --public \
    -d "{tool-name} with results"
```

### Share multiple outputs

```bash
gh gist create \
    outputs/report.md \
    outputs/chart.png \
    --public \
    -d "Analysis results: {description}"
```

## Response Format

After creating the gist, always return:

```
✅ Gist created: {gist-url}

Files included:
- {file1}
- {file2}
```

## Tips

- Use `--public` for shareable links (default)
- Use `--private` for sensitive experiments
- Description should be concise but searchable
- Include output files when they help demonstrate the tool
