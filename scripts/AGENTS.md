# Build and Development Scripts

**Generated:** 2026-09-03
**Commit:** e401dbc
**Branch:** develop

## OVERVIEW

Local automation + focused CI entry. Git hooks in `scripts/git-hooks/` are **copied** (not symlinked) into `.git/hooks/`.

## STRUCTURE

```
scripts/
├── setup-git-hooks.sh
├── run_platform_gates.py       # Focused CI entry
├── benchmark_remediation.py
├── complexity-analysis.sh
├── profile-server.sh
├── test_anthropic_*.py
├── test_cats_dogs.py
└── git-hooks/
    ├── pre-push                # typecheck + lint --all; no pytest
    └── README.md
```

## WHERE TO LOOK

| Script | Purpose |
|--------|---------|
| `setup-git-hooks.sh` | Copy hooks; skip `*.sh` / `*.md` / `*.txt` |
| `run_platform_gates.py` | `--focused` gates; `main() -> int` |
| `git-hooks/pre-push` | `./start-type-check.sh` then `./start-lint.sh --all` |
| `complexity-analysis.sh` | `.reports/complexity-report.txt` |

## CONVENTIONS

```bash
./scripts/setup-git-hooks.sh
./start-type-check.sh || exit 1
./start-lint.sh --all || exit 1
```

- Edit `scripts/git-hooks/`, then re-run setup. `.git/hooks/` is not VCS.
- `--all` **mutates**. `--check` is read-only.
- ANN applies to scripts: `main() -> int`.
- Root wrappers: `./start-type-check.sh`, `./start-lint.sh`, `./start-unittest.sh`.

## ANTI-PATTERNS

- Never edit `.git/hooks/` directly.
- Do not add full lint/type/coverage CI. `.github/workflows/platform-storage.yml` is the only workflow.
- Never treat pre-push lint as `--check` — `--all` writes files.
- Pre-push still no pytest.

## NOTES

`git push --no-verify` exists but is not the workflow. After hook edits, re-run setup.
