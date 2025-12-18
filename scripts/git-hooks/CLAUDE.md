# scripts/git-hooks/

This directory contains version-controlled git hooks for the OpenMemoriesMCP project.

## Structure

```
git-hooks/
├── README.md              # Documentation for hooks
└── pre-push               # Pre-push validation hook
```

## Purpose

Git hooks automate quality checks at specific points in the git workflow. By storing hooks here (not in `.git/hooks/`), we can:
- Version control the hooks
- Share them across the team
- Update them for everyone through git
- Ensure consistent validation for all developers

## Available Hooks

### pre-push

**When it runs**: Before `git push` sends commits to remote repository

**What it does**:
1. Runs type checking with `./start-type-check.sh`
2. Runs linting with `./start-lint.sh --all` (check + fix + format)
3. Blocks push if either fails

**Why**:
- Prevents pushing code with type errors
- Ensures all pushed code meets style standards
- Catches issues before CI/CD runs
- Maintains high code quality in remote branches

**Bypassing** (not recommended):
```bash
git push --no-verify
```

Only bypass for emergencies (hotfixes, urgent fixes) and fix issues immediately after.

## Installation

### Automatic (Recommended)
```bash
./scripts/setup-git-hooks.sh
```

This copies all hooks to `.git/hooks/` and makes them executable.

### Manual
```bash
cp scripts/git-hooks/pre-push .git/hooks/pre-push
chmod +x .git/hooks/pre-push
```

### Verification
```bash
# Check if installed
ls -la .git/hooks/pre-push

# Should show executable permissions (-rwxr-xr-x)
```

## Hook Behavior

### pre-push Flow

```
git push
    ↓
pre-push hook triggered
    ↓
Step 1: Type checking (./start-type-check.sh)
    ├─ Success → Continue
    └─ Failure → Abort push, show error
        ↓
Step 2: Linting (./start-lint.sh --all)
    ├─ Success → Allow push
    └─ Failure → Abort push, show error
```

### Success Output
```
🔒 Pre-Push Hook: Running validation checks...
===========================================

Step 1/2: Running type check...
✅ Type checking passed

Step 2/2: Running linting (check, fix, format)...
✅ Linting passed

🎉 All validation checks passed! Proceeding with push...
```

### Failure Output
```
🔒 Pre-Push Hook: Running validation checks...
===========================================

Step 1/2: Running type check...
❌ Type checking failed!
💡 Please fix type errors before pushing
💡 Run: ./start-type-check.sh
```

## Hook Development

### Code Structure
```bash
#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
# ...

# Get repository root (important for scripts to work from any directory)
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT" || exit 1

# Run checks
if ! ./some-check.sh; then
    echo -e "${RED}❌ Check failed${NC}"
    exit 1
fi

# Success
exit 0
```

### Exit Codes
- `exit 0` - Hook passes, git operation continues
- `exit 1` - Hook fails, git operation aborted

### Testing Hooks

Before committing hook changes:

1. **Install locally**: `./scripts/setup-git-hooks.sh`
2. **Test success case**: Make a valid commit and push to test branch
3. **Test failure case**: Introduce type error or linting issue, try to push
4. **Verify output**: Check colors, messages, error handling
5. **Test from subdirectory**: `cd tests && git push` (should work)

## Adding New Hooks

Git supports many hook types:

### Client-side Hooks
- `pre-commit` - Before commit is created
- `prepare-commit-msg` - Before commit message editor opens
- `commit-msg` - After commit message is entered
- `post-commit` - After commit is created
- `pre-rebase` - Before rebase
- `post-merge` - After merge
- `pre-push` - Before push (implemented)

### Server-side Hooks
- `pre-receive` - Before refs are updated
- `update` - Before each ref is updated
- `post-receive` - After refs are updated

### To Add a New Hook:

1. **Create the hook file**:
```bash
touch scripts/git-hooks/pre-commit
chmod +x scripts/git-hooks/pre-commit
```

2. **Implement the hook**:
```bash
#!/bin/bash
# pre-commit - Run tests before commit

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${BLUE}🔒 Pre-Commit Hook: Running tests...${NC}"

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT" || exit 1

if ! ./start-unittest.sh --quick; then
    echo -e "${RED}❌ Tests failed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Tests passed${NC}"
exit 0
```

3. **Test locally**: `./scripts/setup-git-hooks.sh`

4. **Document**:
   - Add to this CLAUDE.md
   - Update `README.md`
   - Note when it runs and what it checks

5. **Commit and notify team**:
```bash
git add scripts/git-hooks/pre-commit
git commit -m "Add pre-commit hook for running tests"
git push

# Tell team: "Run ./scripts/setup-git-hooks.sh to install new pre-commit hook"
```

## Hook Performance

### Current Performance
- Type checking: ~5-10 seconds
- Linting: ~5-10 seconds
- **Total**: ~10-20 seconds per push

### Optimization Strategies

If hooks become too slow:

1. **Cache results**: Skip if no Python files changed
```bash
# Check if Python files changed
if ! git diff --cached --name-only | grep -q "\.py$"; then
    echo "No Python files changed, skipping checks"
    exit 0
fi
```

2. **Run only on changed files**:
```bash
CHANGED_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep "\.py$" || true)
if [ -n "$CHANGED_FILES" ]; then
    uv run mypy $CHANGED_FILES
fi
```

3. **Parallel execution**:
```bash
./start-type-check.sh &
TYPE_CHECK_PID=$!

./start-lint.sh &
LINT_PID=$!

wait $TYPE_CHECK_PID || exit 1
wait $LINT_PID || exit 1
```

4. **Make hooks optional**:
```bash
if [ "$SKIP_HOOKS" = "1" ]; then
    echo "⚠️  Hooks skipped (SKIP_HOOKS=1)"
    exit 0
fi
```

## Troubleshooting

### Hook Not Running
```bash
# Check if installed
ls -la .git/hooks/pre-push

# If not found, install
./scripts/setup-git-hooks.sh
```

### Hook Failing Incorrectly
```bash
# Run manually to see detailed output
.git/hooks/pre-push

# Or run underlying scripts
./start-type-check.sh
./start-lint.sh --all
```

### Hook Permissions Error
```bash
# Make executable
chmod +x .git/hooks/pre-push
```

### Can't Find Scripts
```bash
# Hook should cd to repo root
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT" || exit 1

# Then scripts work from anywhere
```

## Best Practices

1. **Keep hooks fast**: Developers will bypass slow hooks
2. **Provide clear output**: Use colors and clear messages
3. **Handle errors gracefully**: Show helpful error messages
4. **Make idempotent**: Safe to run multiple times
5. **Test thoroughly**: Both success and failure cases
6. **Document**: Update README.md and CLAUDE.md
7. **Get feedback**: Ask team if hooks are helpful or annoying
8. **Version control**: Always commit hook changes
9. **Communicate changes**: Tell team when hooks are updated

## Skipping Hooks

Use `--no-verify` sparingly:

**When it's OK**:
- Emergency hotfix (fix immediately after)
- Hook is broken (report and fix)
- Testing push behavior

**When it's NOT OK**:
- "I don't want to fix the errors" (bad practice)
- "The hook is slow" (optimize hook instead)
- Regular workflow (fix the root cause)

```bash
# Emergency only
git push --no-verify

# Then immediately
./start-type-check.sh  # Fix issues
./start-lint.sh --all  # Fix issues
git add .
git commit -m "Fix issues from previous commit"
git push  # Without --no-verify
```
