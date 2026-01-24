Build whatever user asks you with techstack user wants . Test if code works (actually run the code and verify results).

Example:

(default python, uv packaga manager)
 (default: quant reserach)

Workflow

Plan → Work → Review → Compound → Repeat
Command	Purpose
/workflows:plan	Turn feature ideas into detailed implementation plans
/workflows:work	Execute plans with worktrees and task tracking
/workflows:review	Multi-agent code review before merging
/workflows:compound	Document learnings to make future work easier
Each cycle compounds: plans inform future plans, reviews catch more issues, patterns get documented.

Philosophy

Each unit of engineering work should make subsequent units easier—not harder.

Traditional development accumulates technical debt. Every feature adds complexity. The codebase becomes harder to work with over time.

Compound engineering inverts this. 80% is in planning and review, 20% is in execution:

Plan thoroughly before writing code
Review to catch issues and capture learnings
Codify knowledge so it's reusable
Keep quality high so future changes are easy