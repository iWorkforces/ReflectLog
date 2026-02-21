# Build and Development Scripts

**Generated:** 2026-02-21
**Commit:** 4c3af26
**Branch:** develop

## OVERVIEW

Custom development scripts for type checking, linting, profiling, and git hooks. No CI/CD - all automation via these scripts.

## STRUCTURE

```
scripts/
├── setup-git-hooks.sh               # Install VCS-controlled hooks
├── complexity-analysis.sh           # Code complexity metrics
├── profile-server.sh                # Server performance profiling
└── git-hooks/
    └── pre-push                     # Type-check + lint enforcement
```

## WHERE TO LOOK

| Script | Purpose |
|--------|---------|
| setup-git-hooks.sh | Link scripts/git-hooks to .git/hooks |
| git-hooks/pre-push | Run type-check + lint before push |
| complexity-analysis.sh | Cyclomatic complexity report |
| profile-server.sh | cProfile server startup |

## KEY PATTERNS

### Git Hook Installation
```bash
./scripts/setup-git-hooks.sh
# Links scripts/git-hooks/* to .git/hooks/*
```

### Pre-Push Enforcement
```bash
# scripts/git-hooks/pre-push
./start-type-check.sh || exit 1
./start-lint.sh --all || exit 1
```

### Complexity Analysis
```bash
./scripts/complexity-analysis.sh
# Uses radon for cyclomatic complexity
# Reports: A (simple) to F (unmaintainable)
```

## ANTI-PATTERNS

- Never modify hooks in .git/hooks directly - edit scripts/git-hooks/
- Never skip pre-push hook (enforced)
- Never add new scripts without color output

## NOTES

- **No CI/CD**: All validation via git hooks
- **VCS-controlled**: Hooks tracked in git
- **Color-coded**: All scripts use ANSI colors
