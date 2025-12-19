# scripts/

This directory contains development automation scripts for the CCMemoriesMCP project.

## Structure

```
scripts/
├── setup-git-hooks.sh          # Install git hooks from git-hooks/
├── test_anthropic_add_flow.py  # Test Anthropic provider add workflow
├── test_anthropic_provider.py  # Test Anthropic provider implementation
├── test_cats_dogs.py           # Test cats/dogs memory scenario
└── git-hooks/                  # Version-controlled git hooks
    ├── README.md
    └── pre-push                # Pre-push validation hook
```

## Purpose

Provides reusable scripts for:
- Setting up development environment
- Installing git hooks
- Project-specific automation tasks

## setup-git-hooks.sh

**Purpose**: Copy git hooks from `scripts/git-hooks/` to `.git/hooks/` and make them executable.

**Why this exists**: Git hooks in `.git/hooks/` are not version-controlled. By storing hooks in `scripts/git-hooks/`, we can share them across the team and update them through git.

**Usage**:
```bash
./scripts/setup-git-hooks.sh
```

**What it does**:
1. Checks if script is run from repository root
2. Verifies `.git/hooks/` directory exists
3. Copies all executable files from `scripts/git-hooks/` to `.git/hooks/`
4. Makes them executable (`chmod +x`)
5. Skips non-hook files (README.md, etc.)

**When to run**:
- After cloning the repository
- After pulling updates that modify hooks
- When hooks aren't working as expected

**Read the implementation**: `scripts/setup-git-hooks.sh` for details

## Test Scripts

These Python scripts are used for testing specific features and providers.

### test_anthropic_add_flow.py

**Purpose**: Test the add workflow using the Anthropic provider.

**Usage**:
```bash
uv run python scripts/test_anthropic_add_flow.py
```

**What it tests**:
- Memory add operations with Anthropic provider
- Smart replacement detection flow
- Provider-specific configuration

### test_anthropic_provider.py

**Purpose**: Test the Anthropic provider implementation directly.

**Usage**:
```bash
uv run python scripts/test_anthropic_provider.py
```

**What it tests**:
- `AnthropicRerankerProvider` functionality
- `AnthropicReplacementProvider` functionality
- Claude SDK integration

### test_cats_dogs.py

**Purpose**: Test a realistic memory replacement scenario.

**Usage**:
```bash
uv run python scripts/test_cats_dogs.py
```

**What it tests**:
- Smart replacement with contradicting memories (e.g., "I like cats" → "I like dogs now")
- Memory archival behavior
- Confidence threshold handling

## git-hooks/

This subdirectory stores version-controlled git hooks. See `git-hooks/CLAUDE.md` for details.

## Adding New Scripts

When adding new automation scripts:

1. **Create the script** in this directory
2. **Make it executable**: `chmod +x scripts/your-script.sh`
3. **Add shebang**: Start with `#!/bin/bash` (or appropriate interpreter)
4. **Add documentation**: Include usage comments at top of file
5. **Follow conventions**:
   - Use `set -e` to exit on errors
   - Add colored output for better UX
   - Include help text (`--help` flag)
   - Validate prerequisites (check for required tools)
6. **Document here**: Add entry in this CLAUDE.md
7. **Test**: Run in clean environment to verify

## Script Conventions

### Colors
```bash
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}✅ Success${NC}"
echo -e "${RED}❌ Error${NC}"
```

### Error Handling
```bash
set -e  # Exit on error

# Check prerequisites
if ! command -v uv &> /dev/null; then
    echo -e "${RED}❌ uv not found${NC}"
    exit 1
fi
```

### Help Text
```bash
show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Description of what the script does.

Options:
  --option1     Description
  --help, -h    Show this help

Examples:
  $(basename "$0") --option1 value
EOF
}
```

### Repository Root Detection
```bash
# Get repository root
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)

if [ -z "$REPO_ROOT" ]; then
    echo "Error: Not in a git repository"
    exit 1
fi

cd "$REPO_ROOT" || exit 1
```

## Common Use Cases

### Running Development Tools

These scripts are at project root (not in `scripts/`):
- `./start-lint.sh` - Code linting
- `./start-type-check.sh` - Type checking
- `./start-unittest.sh` - Unit tests
- `./start-ccmemories-mcp-server.sh` - Start MCP server

See root `CLAUDE.md` for details on these scripts.

### Setting Up Development Environment

```bash
# Clone repository
git clone <repo-url>
cd CCMemoriesMCP

# Install git hooks
./scripts/setup-git-hooks.sh

# Install dependencies
uv sync

# Run tests
./start-unittest.sh
```

## Maintenance

### Updating Git Hooks

1. Modify hooks in `scripts/git-hooks/`
2. Test locally: `./scripts/setup-git-hooks.sh`
3. Commit and push changes
4. Notify team to run: `./scripts/setup-git-hooks.sh`

### Script Dependencies

Scripts may depend on:
- `git` - Version control
- `bash` - Shell interpreter
- `uv` - Python package manager
- Project-specific tools (ruff, mypy, pytest)

Always check for dependencies before running:
```bash
command -v git &> /dev/null || { echo "git not found"; exit 1; }
```

## Best Practices

1. **Make scripts idempotent**: Safe to run multiple times
2. **Validate inputs**: Check arguments and environment
3. **Provide feedback**: Use colors and clear messages
4. **Handle errors gracefully**: Don't leave system in bad state
5. **Document thoroughly**: Help text and comments
6. **Test in isolation**: Run in clean environment
7. **Follow conventions**: Use existing scripts as templates
