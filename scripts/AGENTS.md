# Build and Development Scripts

**Generated:** 2026-08-30  **Commit:** 062b44f  **Branch:** develop

## OVERVIEW
Local automation only. **No CI.** Git hooks live in `scripts/git-hooks/` and are **copied** (not symlinked) into `.git/hooks/`.

## STRUCTURE
```
scripts/
├── setup-git-hooks.sh          # cp scripts/git-hooks/* → .git/hooks
├── complexity-analysis.sh      # radon / mccabe
├── profile-server.sh           # cProfile
├── test_anthropic_*.py         # Ad-hoc provider scripts
├── test_cats_dogs.py
└── git-hooks/
    ├── pre-push                # typecheck + lint --all
    └── README.md
```

## WHERE TO LOOK
| Script | Purpose |
|--------|---------|
| `setup-git-hooks.sh` | Copy hooks; skip `*.sh` / `*.md` / `*.txt` |
| `git-hooks/pre-push` | `./start-type-check.sh` then `./start-lint.sh --all` |
| `complexity-analysis.sh` | `.reports/complexity-report.txt` |
| `profile-server.sh` | `/tmp/profile-stats.prof` |

## CONVENTIONS
```bash
./scripts/setup-git-hooks.sh
# pre-push (mutates working tree via ruff --all):
./start-type-check.sh || exit 1
./start-lint.sh --all || exit 1
```
- Edit `scripts/git-hooks/`, then re-run setup. `.git/hooks/` is not VCS.
- `--all` **mutates** (check + fix + format). `--check` is read-only at repo root.
- Root wrappers: `./start-type-check.sh`, `./start-lint.sh`, `./start-unittest.sh`.

## ANTI-PATTERNS
- Never edit `.git/hooks/` directly.
- Never add GitHub Actions / CI.
- Never treat pre-push lint as `--check` — `--all` writes files.
- Ban `getattr` / `optional_attr()` / `type(obj).__dict__` in ad-hoc scripts.

## NOTES
`git push --no-verify` exists but is not the workflow. After hook edits, everyone must re-run setup. Pre-push does not run pytest.
