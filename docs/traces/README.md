# Traces

One file per bug or feature, readable cold: how it was found or scoped, what
was tried, what worked, what did not, and how it was verified. Name them
`YYYY-MM-DD-short-name.md`. Copy the skeleton below.

```markdown
# <bug|feature>: <title>
- Status: open | done | abandoned      - Opened: <date>      - Model/session: <id>

## Found / scoped
How it surfaced; the exact symptom or the acceptance criterion.

## Tried
- <attempt> -> <result, with the number or error text>

## Worked
What was changed (commit ids) and the root cause in one sentence.

## Verified
The commands run after the change and their output. Counts, not just exit codes.

## Left open
Anything not fixed, and why.
```

| Trace | Kind | Status |
|---|---|---|
| [2026-09-24-csp-blanked-frontend](2026-09-24-csp-blanked-frontend.md) | bug | done |
| [2026-09-29-field-eval-review-routing-scoped-model](2026-09-29-field-eval-review-routing-scoped-model.md) | feature (Phase B) | open |
