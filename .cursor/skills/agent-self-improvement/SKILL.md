---
name: agent-self-improvement
description: Appends agent self-journal entries and creates or updates project skills when repeated workflows or mistakes are discovered. Use after substantive sessions, when documenting agent mistakes, process improvements, or missing playbooks.
disable-model-invocation: true
---

# Agent self-improvement

## Two mechanisms

| Mechanism      | Location                                                            | Purpose                              |
| -------------- | ------------------------------------------------------------------- | ------------------------------------ |
| Self-journal   | [docs/agents/self-journal.md](../../../docs/agents/self-journal.md) | Mistakes, fixes, conventions learned |
| Project skills | [.cursor/skills/](../)                                              | Repeatable multi-step playbooks      |

**Rules** (`.cursor/rules/`) = always-on constraints. **Skills** = on-demand workflows. Do not duplicate rules into skills.

## “Remember this” (user key phrase)

When the user says **remember this** (or close variants: “don’t forget”, “always do X”):

1. **Persist immediately** — do not only acknowledge in chat.
2. **Choose mechanism:**
   - Repeated mistake / convention → prepend [self-journal.md](../../../docs/agents/self-journal.md) **and** add or update an always-on rule in `.cursor/rules/` when the constraint should apply every session.
   - Repeatable workflow → new or updated project skill + skills README row.
3. **Confirm in chat** what was written and where (rule path, journal id).

Treat “remember this” as explicit permission to update tracked agent docs without a separate commit request (user still controls git commits).

## When to prepend the journal

After substantive sessions when you discover:

- A mistake that wasted time or misled conclusions
- A fix or workaround that should become default
- A user-established convention
- A repeated workflow missing from skills

### Entry format

```markdown
### JRN-YYYYMMDD-NN: Short title

| Field            | Value                                            |
| ---------------- | ------------------------------------------------ |
| **Timestamp**    | YYYY-MM-DD HH:MM:SS (system clock at write time) |
| **Category**     | mistake \| fix \| convention \| skill-gap        |
| **Summary**      | What happened                                    |
| **Action taken** | Code, doc, skill, or rule change                 |
| **Related**      | EXP-…, NOTE-…, skill name                        |
```

Insert at the **top** of `## Entries` in [self-journal.md](../../../docs/agents/self-journal.md). Increment `NN` per day. Link research IDs when relevant.

## When to add or update a skill

Trigger: same workflow requested twice, or journal entry category `skill-gap`.

1. Create `.cursor/skills/<name>/SKILL.md` (YAML frontmatter, third-person description, under 500 lines)
2. Add row to [skills README](../README.md)
3. Optional routing hint in [AGENTS.md](../../../AGENTS.md)
4. Prepend journal entry noting the skill-gap closure

Omit `disable-model-invocation` only when the skill should auto-apply from task description (e.g. tide overfit).

## Session-end checklist

- [ ] Research logged (`EXP-…` / `NOTE-…`) per [research-session-workflow](../research-session-workflow/SKILL.md)
- [ ] Process learnings → self-journal
- [ ] New repeated workflow → new skill + index row
- [ ] Broken index links → fix immediately (skill-gap)

## Do not

- Store research findings in the journal (use EXPERIMENT_LOG / DISCUSSION_NOTES)
- Create skills in `~/.cursor/skills-cursor/` (Cursor internal)
