# Project instructions

Start of every session: `git fetch`, then read [HANDOVER.md](HANDOVER.md). Treat it
as testimony from the last session, not as fact; verify what you rely on.

Before changing code, read the documents that bound the change:

- [docs/CONSTRAINTS.md](docs/CONSTRAINTS.md): what you must never or always do here.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/FLOW.md](docs/FLOW.md): the map and the request path.
- [docs/DECISIONS.md](docs/DECISIONS.md): settled arguments; do not reopen one without new evidence.

Working rules that come from the field guide this project follows:

1. One logical change per request; explain the plan before implementing it.
2. Comment non-obvious logic as you write it: what it is for and what depends on it.
3. Log every meaningful decision in `docs/DECISIONS.md` with the model and session that made it.
4. Keep one trace per bug or feature in `docs/traces/`.
5. Done means the commands in [docs/TEST_CHECKLIST.md](docs/TEST_CHECKLIST.md) were run after the change
   and their output read (counts, not only exit codes). Otherwise say "not verified".
6. Undo plan for risky edits lives in [docs/ROLLBACK.md](docs/ROLLBACK.md); check it before large changes.
7. End every session by updating `HANDOVER.md`: what was done, what is left, what to watch for.

Never add AI or assistant attribution to commits, PRs, tags or releases (Pranav is the sole developer of record).
