---
name: retrospective
description: Generate a research retrospective document at the end of research sessions. Analyzes what went well/badly/ugly, reviews skill utilization, and generates actionable improvements. Use after completing research tasks, especially before final commits on feature branches.
---

# Research Retrospective Skill

Generate structured retrospective documents to compound learnings from research sessions.

## When to Use

- After completing a research task
- Before the final commit on a feature branch
- When prompted by the PostToolUse hook after commits
- When you want to document lessons learned from a session

## Workflow

### Phase 1: Context Gathering

Gather context about the current session:

1. **Identify the branch:** `git rev-parse --abbrev-ref HEAD`
2. **Review recent commits:** `git log --oneline -20`
3. **Check changed files:** `git diff --stat HEAD~5..HEAD` (or since branch divergence)
4. **Identify the research task:** Infer from branch name, commit messages, and modified files

### Phase 2: Self-Reflection

Answer these questions based on the session:

**What Went Well?**
- Which approaches worked on the first try?
- What tools/libraries were effective?
- What decisions saved time?

**What Went Badly?**
- What required multiple debugging iterations?
- What assumptions were wrong?
- What took longer than expected?

**What Was Ugly?**
- What code needs refactoring?
- What processes were inefficient?
- What hacks or workarounds were used?

### Phase 3: Skills Analysis

Review skill utilization:

1. List skills that were used and rate their effectiveness
2. Identify skills that SHOULD have been used but weren't
3. Note why skills weren't used (forgot, didn't know, seemed overkill)

**Key skills to evaluate:**
- `/context7` - Was library documentation checked?
- `/qrd` - Was a structured specification created?
- `/workflows:plan` - Was planning done before coding?
- `/workflows:review` - Was code reviewed?
- `Explore` agent - Was it used for codebase questions?
- `data/samples/` - Was cached data used?

### Phase 4: Document Generation

Generate the retrospective document with this structure:

```markdown
# [Task Name]: Process Retrospective

**Date:** YYYY-MM-DD
**Branch:** [branch-name]
**Task:** [Brief description]
**Outcome:** [Success/Partial/Failed - brief summary]

---

## Executive Summary

| Aspect | Rating | Notes |
|--------|--------|-------|
| Research Quality | /10 | [Brief note] |
| Code Quality | /10 | [Brief note] |
| Time Efficiency | /10 | [Brief note] |
| Skill Utilization | /10 | [Brief note] |
| Final Results | /10 | [Brief note] |

---

## What Went Well

### 1. [Category]
[Description with specific examples]

### 2. [Category]
[Description with specific examples]

---

## What Went Badly

### 1. [Issue]
[What happened, root cause, time wasted]

### 2. [Issue]
[What happened, root cause, time wasted]

---

## What Was Ugly

### 1. [Problem]
[Description and why it's ugly]

---

## Skills Analysis

### Skills Used

| Skill | Usage | Effectiveness |
|-------|-------|---------------|
| [Skill] | [What for] | [Rating + notes] |

### Skills That Should Have Been Used

| Skill | Why Not Used | What Was Missed |
|-------|--------------|-----------------|
| [Skill] | [Reason] | [Impact] |

---

## Code & Repo Metrics

| Metric | Value |
|--------|-------|
| Execution time (approx) | |
| Files changed | |
| Files added | |
| Files deleted | |
| LOC added | |
| LOC removed | |
| Test pass rate | |
| Lint issues | |
| Dependency changes | |
| Errors encountered | |

---

## Structural & Process Issues

Identify problems with:
- **Repo structure:** Misplaced files, unclear organization
- **Instructions:** Missing or unclear documentation
- **Automation:** Missing scripts or workflows
- **Redundant code:** Duplicated logic that should be shared
- **Missing abstractions:** Repeated patterns that should be modular

[List specific issues found]

---

## Recommended Skill Usage Next Run

Explicit checklist for next similar task:

- [ ] [Tool/script to prefer]
- [ ] [Cached data to use]
- [ ] [Shared module to leverage]
- [ ] [Workflow to follow]
- [ ] [Documentation to check]

---

## Actionable Improvements

### Immediate (Do Now)
1. [Specific action]
2. [Specific action]

### Short-Term (This Week)
1. [Specific action]
2. [Specific action]

### Long-Term (This Month)
1. [Specific action]
2. [Specific action]

---

## Lessons Learned

### For Future [Domain] Research
1. [Lesson]
2. [Lesson]

### For Claude Code Usage
1. [Lesson]
2. [Lesson]

---

## Summary

| Category | Key Takeaway |
|----------|--------------|
| **Research** | [Main insight] |
| **Code** | [Main insight] |
| **Skills** | [Main insight] |
| **Process** | [Main insight] |

---

## Final Self-Score

| Dimension | Score | Notes |
|-----------|-------|-------|
| Speed | /10 | |
| Code Quality | /10 | |
| Tool Usage | /10 | |
| Repo Alignment | /10 | |
| **Overall** | **/10** | |

**Justification (2-3 lines max):**
[Brief explanation of scores and main areas for improvement]
```

### Phase 5: Save Document

Save to: `output/{branch-name}/retrospective_{YYYYMMDD_HHMMSS}.md`

```bash
# Get branch name (sanitized for filesystem)
branch=$(git rev-parse --abbrev-ref HEAD | tr '/' '-')

# Create output directory
mkdir -p "output/${branch}"

# Generate timestamp
timestamp=$(date +%Y%m%d_%H%M%S)

# Save file
# output/${branch}/retrospective_${timestamp}.md
```

## Output Location

- **Directory:** `output/{branch-name}/`
- **Filename:** `retrospective_{YYYYMMDD_HHMMSS}.md`
- **Branch name sanitization:** Replace `/` with `-`

## Behavioral Rules

When generating retrospectives:
- Be critical and honest - the goal is learning, not looking good
- Prioritize engineering improvements over process observations
- Prefer reuse over new code - identify what should have been reused
- Assume future runs should be faster and simpler
- Do not write motivational text or filler
- Do not repeat context already in the report
- Keep it technical and precise
- Quantify everything possible

## Tips for Effective Retrospectives

1. **Be honest** - The goal is learning, not looking good
2. **Be specific** - "Debugging took long" is useless; "yfinance API change cost 30min" is actionable
3. **Quantify when possible** - Time wasted, lines of code deleted, iterations required
4. **Focus on actionable items** - Each improvement should be specific enough to implement
5. **Link to existing patterns** - Reference `docs/memory/patterns.md` if applicable
6. **Update memory files** - If learnings are reusable, add to `docs/memory/`

## Integration with Other Skills

- After `/retrospective`, consider running `/workflows:compound` to formalize learnings
- Update `.claude/rules/` if new workflow patterns emerge
- Add reusable code to `shared/` if identified during retrospective
