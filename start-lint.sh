#!/bin/bash

# CCProxy - Code Linting Script
# This script uses ruff to scan and fix all Python files for linting issues

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PYTHON_FILES_PATTERN="*.py"
RUFF_CONFIG_FILE="pyproject.toml"
VENV_DIR="venv"

echo -e "${BLUE}🔍 ReflectLog - Code Linting${NC}"
echo -e "${BLUE}=====================================${NC}"
echo ""

uv_run() {
    command uv run --frozen --no-sync "$@"
}

# Function to require uv without installing it
check_uv() {
    if ! command -v uv &> /dev/null; then
        echo -e "${RED}❌ uv is required but was not found${NC}"
        echo -e "${YELLOW}Install uv separately: https://docs.astral.sh/uv/getting-started/installation/${NC}"
        exit 1
    fi

    echo -e "${GREEN}✅ uv is available${NC}"
    echo -e "${CYAN}Version: $(uv --version)${NC}"
}

# Function to check if ruff is available in the locked environment
check_ruff() {
    if ! uv_run ruff --version &> /dev/null; then
        echo -e "${RED}❌ ruff is not available in the frozen environment${NC}"
        echo -e "${YELLOW}Prepare the environment separately: uv sync --frozen${NC}"
        exit 1
    fi

    echo -e "${GREEN}✅ ruff is available${NC}"
    echo -e "${CYAN}Version: $(uv_run ruff --version)${NC}"
}

# Function to upgrade ruff to latest version
upgrade_ruff() {
    echo -e "${BLUE}🔄 Upgrading ruff to latest version...${NC}"

    # Use uv to upgrade ruff if available, otherwise fall back to pip
    if command -v uv &> /dev/null; then
        echo -e "${CYAN}Upgrading ruff via uv...${NC}"
        uv pip install --upgrade ruff
    elif [[ "$VIRTUAL_ENV" != "" ]]; then
        echo -e "${GREEN}✅ Using active virtual environment${NC}"
        pip install --upgrade ruff
    elif [ -d "$VENV_DIR" ]; then
        echo -e "${YELLOW}🔧 Activating virtual environment...${NC}"
        source "$VENV_DIR/bin/activate"
        pip install --upgrade ruff
    else
        echo -e "${YELLOW}🔧 Upgrading ruff globally...${NC}"
        pip install --user --upgrade ruff
    fi

    echo -e "${GREEN}✅ ruff upgraded to latest version${NC}"
    echo -e "${CYAN}Version: $(uv run ruff --version)${NC}"
    echo ""
}

# Function to find Python files
find_python_files() {
    echo -e "${BLUE}🔍 Finding Python files...${NC}"

    # Find all Python files, excluding common directories to ignore
    PYTHON_FILES=$(find . -name "*.py" \
        -not -path "./venv/*" \
        -not -path "./env/*" \
        -not -path "./.venv/*" \
        -not -path "./build/*" \
        -not -path "./dist/*" \
        -not -path "./.git/*" \
        -not -path "./__pycache__/*" \
        -not -path "./.pytest_cache/*" \
        -not -path "./node_modules/*" \
        -not -path "./tests/*" \
        | sort)

    if [ -z "$PYTHON_FILES" ]; then
        echo -e "${YELLOW}⚠️  No Python files found${NC}"
        exit 0
    fi

    FILE_COUNT=$(echo "$PYTHON_FILES" | wc -l)
    echo -e "${GREEN}✅ Found $FILE_COUNT Python files${NC}"
    echo ""
}

# Function to show ruff configuration
show_config() {
    echo -e "${BLUE}⚙️  Ruff Configuration:${NC}"

    if [ -f "$RUFF_CONFIG_FILE" ]; then
        echo -e "${GREEN}✅ Using configuration from $RUFF_CONFIG_FILE${NC}"

        # Show relevant ruff configuration if it exists
        if grep -q "\[tool.ruff\]" "$RUFF_CONFIG_FILE" 2>/dev/null; then
            echo -e "${CYAN}Configuration preview:${NC}"
            sed -n '/\[tool\.ruff\]/,/^\[/p' "$RUFF_CONFIG_FILE" | head -20
        fi
    else
        echo -e "${YELLOW}⚠️  No ruff configuration found, using defaults${NC}"
        echo -e "${CYAN}Default rules: E, W, F (pycodestyle errors, warnings, pyflakes)${NC}"
    fi
    echo ""
}

# Function to run ruff check (dry run)
run_check() {
    echo -e "${BLUE}🔍 Running ruff check (dry run)...${NC}"
    echo ""

    if uv_run ruff check reflectlog tests scripts --output-format=full \
        && uv_run ruff format --check reflectlog tests scripts; then
        echo ""
        echo -e "${GREEN}✅ No linting issues found!${NC}"
        return 0
    else
        echo ""
        echo -e "${YELLOW}⚠️  Linting issues detected${NC}"
        return 1
    fi
}

# Function to run ruff check with statistics
run_check_stats() {
    echo -e "${BLUE}📊 Linting Statistics:${NC}"
    echo ""

    # Get statistics by rule (only reflectlog directory, tests excluded)
    echo -e "${CYAN}Issues by rule:${NC}"
    uv_run ruff check reflectlog tests scripts --output-format=concise | cut -d: -f4 | cut -d' ' -f2 | sort | uniq -c | sort -nr || true
    echo ""

    # Get statistics by file
    echo -e "${CYAN}Files with issues:${NC}"
    uv_run ruff check reflectlog tests scripts --output-format=concise | cut -d: -f1 | sort | uniq -c | sort -nr | head -10 || true
    echo ""
}

# Function to fix issues automatically
run_fix() {
    echo -e "${BLUE}🔧 Running ruff fix (automatic fixes)...${NC}"
    echo ""

    # Run ruff with --fix flag (only reflectlog directory, tests excluded)
    if uv run ruff check reflectlog --fix; then
        echo ""
        echo -e "${GREEN}✅ Automatic fixes applied successfully${NC}"
    else
        echo ""
        echo -e "${YELLOW}⚠️  Some issues were fixed, but manual fixes may be needed${NC}"
    fi

    # Show remaining issues
    echo ""
    echo -e "${BLUE}🔍 Checking for remaining issues...${NC}"
    if uv run ruff check reflectlog --output-format=concise; then
        echo -e "${GREEN}✅ All fixable issues have been resolved${NC}"
    else
        echo -e "${YELLOW}⚠️  Some issues require manual attention${NC}"
    fi
}

# Function to replace triple double quotes with triple single quotes
replace_triple_quotes() {
    echo -e "${BLUE}🔄 Replacing triple double quotes with single quotes...${NC}"
    echo ""

    # Find all Python files
    find_python_files

    local count=0
    echo "${PYTHON_FILES}" | while read -r file; do
        if [ -n "${file}" ]; then
            # Check if file contains triple double quotes
            if grep -q '"""' "${file}"; then
                echo -e "${CYAN}Processing: ${file}${NC}"

                # Replace triple double quotes with triple single quotes
                if [[ "${OSTYPE}" == "darwin"* ]]; then
                    # macOS sed syntax
                    sed -i '' 's/"""'/"'''"/g "${file}"
                else
                    # Linux sed syntax
                    sed -i 's/"""'/"'''"/g "${file}"
                fi

                ((count++))
                echo -e "${GREEN}✅ Updated quotes in: ${file}${NC}"
            fi
        fi
    done

    echo ""
    if [ "$count" -gt 0 ]; then
        echo -e "${GREEN}✅ Replaced triple quotes in $count files${NC}"
    else
        echo -e "${YELLOW}ℹ️  No triple double quotes found to replace${NC}"
    fi
}

# Function to format code
run_format() {
    echo -e "${BLUE}🎨 Running ruff format...${NC}"
    echo ""

    # Run ruff format (only reflectlog directory, tests excluded; ignore failures)
    # Quote style is configured in pyproject.toml [tool.ruff.format]
    uv run ruff format reflectlog || true

    # Remove trailing whitespace in all Python files
    find_python_files
    echo -e "${BLUE}🔧 Removing trailing spaces...${NC}"
    if [[ "${OSTYPE}" == "darwin"* ]]; then
        echo "${PYTHON_FILES}" | while read -r file; do
            [ -n "${file}" ] && sed -i '' -E 's/[[:space:]]+$//' "${file}"
        done
    else
        echo "${PYTHON_FILES}" | while read -r file; do
            [ -n "${file}" ] && sed -i -E 's/[[:space:]]+$//' "${file}"
        done
    fi

    echo ""
    echo -e "${GREEN}✅ Code formatting and trailing space removal completed${NC}"
}

# Function to check Python 3.14 migration status
run_migration_check() {
    echo -e "${BLUE}🐍 Python 3.14 Migration Check${NC}"
    echo -e "${BLUE}=======================================${NC}"
    echo ""

    local HAS_ERRORS=0

    # Check 1: Find legacy typing imports (List, Dict, Optional, Union, Tuple - excluding valid ones)
    echo -e "${CYAN}1. Checking for legacy typing imports (List, Dict, Optional, Union, Tuple)...${NC}"
    LEGACY_IMPORTS=$(grep -r "from typing import.*List\|from typing import.*Dict\|from typing import.*Optional\|from typing import.*Union\|from typing import.*Tuple" reflectlog --include="*.py" 2>/dev/null | grep -v "TYPE_CHECKING" | grep -v "Literal\|TypedDict\|Protocol\|Callable\|TypeVar\|Generic\|TypeAlias\|Annotated\|Any" || true)
    if [ -n "$LEGACY_IMPORTS" ]; then
        echo -e "${RED}❌ Found legacy typing imports:${NC}"
        echo "$LEGACY_IMPORTS" | head -10
        HAS_ERRORS=1
    else
        echo -e "${GREEN}✅ No legacy typing imports found${NC}"
    fi
    echo ""

    # Check 2: Find legacy type annotations in code (excluding comments)
    echo -e "${CYAN}2. Checking for legacy type annotations (List[ Dict[ Optional[ Union[ Tuple[ )...${NC}"
    LEGACY_ANNOTATIONS=$(grep -r "List\[\|Dict\[\|Optional\[\|Union\[\|Tuple\[" reflectlog --include="*.py" 2>/dev/null | grep -v "^[^:]*:#" | grep -v "OrderedDict" | grep -v "defaultdict" || true)
    if [ -n "$LEGACY_ANNOTATIONS" ]; then
        echo -e "${RED}❌ Found legacy type annotations:${NC}"
        echo "$LEGACY_ANNOTATIONS" | head -10
        HAS_ERRORS=1
    else
        echo -e "${GREEN}✅ No legacy type annotations found${NC}"
    fi
    echo ""

    # Check 3: Verify only valid typing imports remain
    echo -e "${CYAN}3. Checking for required typing imports...${NC}"
    NEEDED_IMPORTS=$(grep -r "from typing import" reflectlog --include="*.py" 2>/dev/null | grep -v "TYPE_CHECKING" | grep -v "Protocol\|Literal\|Callable\|TypeVar\|Generic\|TypeAlias\|Annotated\|Any" || true)
    if [ -n "$NEEDED_IMPORTS" ]; then
        echo -e "${YELLOW}⚠️  Found typing imports that should be reviewed:${NC}"
        echo "$NEEDED_IMPORTS" | head -10
    else
        echo -e "${GREEN}✅ All typing imports are valid for Python 3.14${NC}"
    fi
    echo ""

    # Check 4: Triple quotes enforcement (only reflectlog directory, tests excluded)
    echo -e "${CYAN}4. Checking for triple double quotes (should be single)...${NC}"
    DOUBLE_QUOTES=$(grep -r '"""' reflectlog --include="*.py" 2>/dev/null | grep -v "^Binary" | head -5 || true)
    if [ -n "$DOUBLE_QUOTES" ]; then
        echo -e "${YELLOW}⚠️  Found triple double quotes (should use single quotes for docstrings):${NC}"
        echo "$DOUBLE_QUOTES" | head -5
    else
        echo -e "${GREEN}✅ No triple double quotes found${NC}"
    fi
    echo ""

    # Check 5: Run ruff UP rules for Python 3.14 suggestions
    echo -e "${CYAN}5. Running ruff pyupgrade rules (UP006, UP007, UP035, UP040)...${NC}"
    UP_ISSUES=$(uv run ruff check reflectlog --select=UP --output-format=concise 2>&1 | grep "UP" || true)
    if [ -n "$UP_ISSUES" ]; then
        echo -e "${YELLOW}⚠️  Python 3.14 upgrade suggestions found:${NC}"
        echo "$UP_ISSUES" | head -20
        # Don't fail on UP rules as they're suggestions
        echo -e "${CYAN}💡 Run '$0 --fix' to apply some automatic fixes${NC}"
    else
        echo -e "${GREEN}✅ No pyupgrade suggestions needed${NC}"
    fi
    echo ""

    if [ $HAS_ERRORS -eq 1 ]; then
        echo -e "${RED}❌ Migration check completed with errors${NC}"
        echo -e "${YELLOW}💡 See .claude/docs/migration-checklist.md for migration steps${NC}"
        return 1
    else
        echo -e "${GREEN}✅ Migration check completed successfully${NC}"
        return 0
    fi
}

# Function to show detailed help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --check       Run linting check only (default)"
    echo "  --fix         Run linting check and apply automatic fixes"
    echo "  --format      Run code formatting (ruff format)"
    echo "  --quotes      Replace triple double quotes with single quotes"
    echo "  --migrate     Check Python 3.14 migration status"
    echo "  --all         Run check, fix, and format"
    echo "  --stats       Show detailed linting statistics"
    echo "  --config      Show ruff configuration"
    echo "  --files       List Python files that will be processed"
    echo "  --install     Install/upgrade ruff"
    echo "  --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Check for issues"
    echo "  $0 --fix             # Fix issues automatically"
    echo "  $0 --format          # Format code"
    echo "  $0 --quotes          # Replace triple double quotes"
    echo "  $0 --migrate         # Check Python 3.14 migration status"
    echo "  $0 --all             # Check, fix, and format"
    echo "  $0 --stats           # Show statistics"
    echo ""
    echo "Environment Variables:"
    echo "  RUFF_CONFIG    Path to ruff configuration file"
    echo "  RUFF_CACHE_DIR Directory for ruff cache"
    echo ""
    echo "Common ruff rules:"
    echo "  E             pycodestyle errors"
    echo "  W             pycodestyle warnings"
    echo "  F             pyflakes"
    echo "  I             isort (import sorting)"
    echo "  N             pep8-naming"
    echo "  UP            pyupgrade (Python 3.14+ modernizations)"
    echo "  B             flake8-bugbear"
    echo "  C4            flake8-comprehensions"
    echo "  SIM           flake8-simplify"
}

# Function to install/upgrade ruff
install_ruff() {
    echo -e "${BLUE}📦 Installing/upgrading ruff...${NC}"

    # Use uv if available, otherwise fall back to pip
    if command -v uv &> /dev/null; then
        echo -e "${CYAN}Installing ruff via uv...${NC}"
        uv pip install --upgrade ruff
    elif [[ "$VIRTUAL_ENV" != "" ]]; then
        echo -e "${GREEN}✅ Using active virtual environment${NC}"
        pip install --upgrade ruff
    elif [ -d "$VENV_DIR" ]; then
        echo -e "${YELLOW}🔧 Activating virtual environment...${NC}"
        source "$VENV_DIR/bin/activate"
        pip install --upgrade ruff
    else
        echo -e "${YELLOW}🔧 Installing ruff globally...${NC}"
        pip install --user --upgrade ruff
    fi

    echo -e "${GREEN}✅ ruff installation completed${NC}"
    echo -e "${CYAN}Version: $(uv run ruff --version)${NC}"
}

# Function to list files
list_files() {
    echo -e "${BLUE}📁 Python files to be processed:${NC}"
    echo ""

    find_python_files

    echo -e "${CYAN}Files:${NC}"
    echo "$PYTHON_FILES" | while read -r file; do
        if [ -n "$file" ]; then
            echo "  $file"
        fi
    done
    echo ""
    echo -e "${GREEN}Total: $FILE_COUNT files${NC}"
}

# Main execution
main() {
    # Parse command line arguments
    ACTION="check"
    SHOW_STATS=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --check)
                ACTION="check"
                shift
                ;;
            --fix)
                ACTION="fix"
                shift
                ;;
            --format)
                ACTION="format"
                shift
                ;;
            --quotes)
                ACTION="quotes"
                shift
                ;;
            --migrate)
                ACTION="migrate"
                shift
                ;;
            --all)
                ACTION="all"
                shift
                ;;
            --stats)
                SHOW_STATS=true
                shift
                ;;
            --config)
                check_uv
                check_ruff
                show_config
                exit 0
                ;;
            --files)
                list_files
                exit 0
                ;;
            --install)
                check_uv
                install_ruff
                exit 0
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                echo -e "${RED}Unknown option: $1${NC}"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done

    # Execute main workflow
    check_uv
    check_ruff
    find_python_files
    show_config

    case $ACTION in
        "check")
            if run_check; then
                if [ "$SHOW_STATS" = true ]; then
                    run_check_stats
                fi
                echo -e "${GREEN}🎉 Linting check completed successfully${NC}"
            else
                if [ "$SHOW_STATS" = true ]; then
                    run_check_stats
                fi
                echo ""
                echo -e "${YELLOW}💡 To fix issues automatically, run: $0 --fix${NC}"
                exit 1
            fi
            ;;
        "fix")
            run_fix
            if [ "$SHOW_STATS" = true ]; then
                run_check_stats
            fi
            echo -e "${GREEN}🎉 Linting fix completed${NC}"
            ;;
        "format")
            run_format
            echo -e "${GREEN}🎉 Code formatting completed${NC}"
            ;;
        "quotes")
            replace_triple_quotes
            echo -e "${GREEN}🎉 Triple quote replacement completed${NC}"
            ;;
        "migrate")
            run_migration_check
            ;;
        "all")
            echo -e "${PURPLE}🚀 Running complete linting workflow...${NC}"
            echo ""

            # Step 1: Check
            echo -e "${BLUE}Step 1: Initial check${NC}"
            run_check || true
            echo ""

            # Step 2: Fix
            echo -e "${BLUE}Step 2: Apply fixes${NC}"
            run_fix
            echo ""

            # Step 3: Format
            echo -e "${BLUE}Step 3: Format code${NC}"
            run_format
            echo ""

            # Step 4: Final check
            echo -e "${BLUE}Step 4: Final verification${NC}"
            if run_check; then
                echo -e "${GREEN}🎉 Complete linting workflow finished successfully${NC}"
            else
                echo -e "${YELLOW}⚠️  Some issues may require manual attention${NC}"
            fi

            if [ "$SHOW_STATS" = true ]; then
                echo ""
                run_check_stats
            fi
            ;;
    esac
}

# Handle Ctrl+C gracefully
trap 'echo -e "\n${YELLOW}Linting interrupted${NC}"; exit 1' INT

# Run main function
main "$@"