#!/bin/bash

# ReflectLog - Type Checking Script
# This script uses ty and pyright to perform static type checking on all Python files

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
TY_CONFIG_FILE="pyproject.toml"

echo -e "${BLUE}🔍 ReflectLog - Type Checking (ty + pyright)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to check if uv is installed
check_uv() {
    if ! command -v uv &> /dev/null; then
        echo -e "${YELLOW}📦 uv not found, installing...${NC}"

        # Install uv using the official installer
        if command -v curl &> /dev/null; then
            echo -e "${CYAN}Installing uv via curl...${NC}"
            curl -LsSf https://astral.sh/uv/install.sh | sh
        elif command -v wget &> /dev/null; then
            echo -e "${CYAN}Installing uv via wget...${NC}"
            wget -qO- https://astral.sh/uv/install.sh | sh
        else
            echo -e "${RED}❌ Neither curl nor wget found. Please install uv manually${NC}"
            echo -e "${YELLOW}Visit: https://docs.astral.sh/uv/getting-started/installation/${NC}"
            exit 1
        fi

        # Source the shell profile to make uv available
        if [ -f "$HOME/.bashrc" ]; then
            source "$HOME/.bashrc"
        elif [ -f "$HOME/.zshrc" ]; then
            source "$HOME/.zshrc"
        fi

        # Add uv to PATH for this session if not already available
        if ! command -v uv &> /dev/null && [ -f "$HOME/.cargo/bin/uv" ]; then
            export PATH="$HOME/.cargo/bin:$PATH"
        fi

        # Verify installation
        if ! command -v uv &> /dev/null; then
            echo -e "${RED}❌ Failed to install uv${NC}"
            echo -e "${YELLOW}Please install uv manually: https://docs.astral.sh/uv/getting-started/installation/${NC}"
            exit 1
        fi
    fi

    echo -e "${GREEN}✅ uv is available${NC}"
    echo -e "${CYAN}Version: $(uv --version)${NC}"
}

# Function to check if ty is installed
check_ty() {
    if ! uv run ty --version &> /dev/null; then
        echo -e "${YELLOW}📦 ty not found, installing...${NC}"
        echo -e "${CYAN}Installing ty via uv...${NC}"
        uv pip install ty

        # Verify installation
        if ! uv run ty --version &> /dev/null; then
            echo -e "${RED}❌ Failed to install ty${NC}"
            echo -e "${YELLOW}Please install ty manually: uv pip install ty${NC}"
            exit 1
        fi
    fi

    echo -e "${GREEN}✅ ty is available${NC}"
    echo -e "${CYAN}Version: $(uv run ty --version)${NC}"
}

check_pyright() {
    if ! uv run pyright --version &> /dev/null; then
        echo -e "${YELLOW}📦 pyright not found, installing...${NC}"
        echo -e "${CYAN}Installing pyright via uv...${NC}"
        uv pip install pyright

        if ! uv run pyright --version &> /dev/null; then
            echo -e "${RED}❌ Failed to install pyright${NC}"
            echo -e "${YELLOW}Please install pyright manually: uv pip install pyright${NC}"
            exit 1
        fi
    fi

    echo -e "${GREEN}✅ pyright is available${NC}"
    echo -e "${CYAN}Version: $(uv run pyright --version)${NC}"
}

# Function to upgrade ty to latest version
upgrade_ty() {
    echo -e "${BLUE}🔄 Upgrading ty to latest version...${NC}"
    echo -e "${CYAN}Upgrading ty via uv...${NC}"
    uv pip install --upgrade ty

    echo -e "${GREEN}✅ ty upgraded to latest version${NC}"
    echo -e "${CYAN}Version: $(uv run ty --version)${NC}"
    echo ""
}

# Function to show ty configuration
show_config() {
    echo -e "${BLUE}⚙️  ty Configuration:${NC}"

    if [ -f "$TY_CONFIG_FILE" ]; then
        echo -e "${GREEN}✅ Using configuration from $TY_CONFIG_FILE${NC}"

        # Show relevant ty configuration if it exists
        if grep -q "\[tool.ty\]" "$TY_CONFIG_FILE" 2>/dev/null; then
            echo -e "${CYAN}Configuration preview:${NC}"
            sed -n '/\[tool\.ty\]/,/^\[tool\.[^t]/p' "$TY_CONFIG_FILE" | head -40
        fi
    else
        echo -e "${YELLOW}⚠️  No ty configuration found, using defaults${NC}"
    fi
    echo ""
}

# Fail closed if getattr() or class-dict probes remain in project sources.
check_no_dynamic_attr_probes() {
    echo -e "${BLUE}🔍 Checking for banned dynamic attribute probes...${NC}"
    if uv run python - <<'PY'
import re
import sys
from pathlib import Path

patterns = (
    re.compile(r"\bggetattr\s*\("),
    re.compile(r"type\([^)]+\)\.__dict__"),
    re.compile(r"\bclass_callable\s*\("),
)
hits: list[str] = []
for root in (Path("reflectlog"), Path("tests")):
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if any(pattern.search(line) for pattern in patterns):
                hits.append(f"{path}:{line_no}:{line}")
if hits:
    print("\n".join(hits))
    sys.exit(1)
PY
    then
        echo -e "${GREEN}✅ No getattr() or type().__dict__ probes found${NC}"
        echo ""
        return 0
    fi
    echo ""
    echo -e "${RED}❌ Dynamic attribute probes are banned. Use typed protocol methods.${NC}"
    return 1
}

# Function to run ty type check
run_type_check() {
    echo -e "${BLUE}🔍 Running ty type check...${NC}"
    echo ""

    # Run ty with detailed output
    if uv run ty check; then
        echo ""
        echo -e "${GREEN}✅ No type errors found!${NC}"
        return 0
    else
        echo ""
        echo -e "${YELLOW}⚠️  Type errors detected${NC}"
        return 1
    fi
}

run_pyright_check() {
    echo -e "${BLUE}🔍 Running pyright type check...${NC}"
    echo ""

    if uv run pyright; then
        echo ""
        echo -e "${GREEN}✅ No pyright type errors found!${NC}"
        return 0
    else
        echo ""
        echo -e "${YELLOW}⚠️  Pyright type errors detected${NC}"
        return 1
    fi
}

# Function to run ty type check with concise output (for CI)
run_type_check_concise() {
    echo -e "${BLUE}🔍 Running ty type check (concise)...${NC}"
    echo ""

    # Run ty with concise output
    if uv run ty check --output-format concise; then
        echo ""
        echo -e "${GREEN}✅ No type errors found!${NC}"
        return 0
    else
        echo ""
        echo -e "${YELLOW}⚠️  Type errors detected${NC}"
        return 1
    fi
}

# Function to show type check statistics
show_stats() {
    echo -e "${BLUE}📊 Type Check Statistics:${NC}"
    echo ""

    # Get statistics by error type
    echo -e "${CYAN}Errors by type:${NC}"
    uv run ty check 2>&1 | grep -E "error\[" | sed 's/.*error\[\([^]]*\)\].*/\1/' | sort | uniq -c | sort -nr || true
    echo ""

    # Get statistics by file
    echo -e "${CYAN}Files with errors:${NC}"
    uv run ty check 2>&1 | grep -E "^[^ ].*\.py:" | cut -d: -f1 | sort | uniq -c | sort -nr | head -10 || true
    echo ""
}

# Function to show detailed help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --check       Run type checking (default)"
    echo "  --concise     Run ty with concise output, then pyright"
    echo "  --stats       Show detailed type error statistics"
    echo "  --config      Show ty configuration"
    echo "  --install     Install/upgrade ty"
    echo "  --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Run type check"
    echo "  $0 --concise          # Run with concise output"
    echo "  $0 --stats            # Show statistics"
    echo ""
    echo "Environment Variables:"
    echo "  TY_LOG              Set log level (debug, info, warn, error)"
    echo ""
}

# Function to install/upgrade ty
install_ty() {
    echo -e "${BLUE}📦 Installing/upgrading ty...${NC}"
    echo -e "${CYAN}Installing ty via uv...${NC}"
    uv pip install --upgrade ty

    echo -e "${GREEN}✅ ty installation completed${NC}"
    echo -e "${CYAN}Version: $(uv run ty --version)${NC}"
}

# Main execution
main() {
    # Parse command line arguments
    ACTION="check"
    SHOW_STATS=false
    OUTPUT_FORMAT="full"

    while [[ $# -gt 0 ]]; do
        case $1 in
            --check)
                ACTION="check"
                shift
                ;;
            --concise)
                ACTION="check"
                OUTPUT_FORMAT="concise"
                shift
                ;;
            --stats)
                SHOW_STATS=true
                shift
                ;;
            --config)
                check_uv
                check_ty
                show_config
                exit 0
                ;;
            --install)
                check_uv
                install_ty
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
    check_ty
    check_pyright
    upgrade_ty
    show_config

    case $ACTION in
        "check")
            GETATTR_EXIT_CODE=0
            if ! check_no_dynamic_attr_probes; then
                GETATTR_EXIT_CODE=1
            fi

            if [ "$OUTPUT_FORMAT" = "concise" ]; then
                TY_EXIT_CODE=0
                PYRIGHT_EXIT_CODE=0

                if ! run_type_check_concise; then
                    TY_EXIT_CODE=1
                fi

                if ! run_pyright_check; then
                    PYRIGHT_EXIT_CODE=1
                fi

                if [ $GETATTR_EXIT_CODE -eq 0 ] && [ $TY_EXIT_CODE -eq 0 ] && [ $PYRIGHT_EXIT_CODE -eq 0 ]; then
                    if [ "$SHOW_STATS" = true ]; then
                        show_stats
                    fi
                    echo -e "${GREEN}🎉 Type checking completed successfully${NC}"
                else
                    if [ "$SHOW_STATS" = true ]; then
                        show_stats
                    fi
                    echo ""
                    echo -e "${YELLOW}💡 Review the errors above and fix type issues${NC}"
                    exit 1
                fi
            else
                TY_EXIT_CODE=0
                PYRIGHT_EXIT_CODE=0

                if ! run_type_check; then
                    TY_EXIT_CODE=1
                fi

                if ! run_pyright_check; then
                    PYRIGHT_EXIT_CODE=1
                fi

                if [ $GETATTR_EXIT_CODE -eq 0 ] && [ $TY_EXIT_CODE -eq 0 ] && [ $PYRIGHT_EXIT_CODE -eq 0 ]; then
                    if [ "$SHOW_STATS" = true ]; then
                        show_stats
                    fi
                    echo -e "${GREEN}🎉 Type checking completed successfully${NC}"
                else
                    if [ "$SHOW_STATS" = true ]; then
                        show_stats
                    fi
                    echo ""
                    echo -e "${YELLOW}💡 Review the errors above and fix type issues${NC}"
                    exit 1
                fi
            fi
            ;;
    esac
}

# Handle Ctrl+C gracefully
trap 'echo -e "\n${YELLOW}Type checking interrupted${NC}"; exit 1' INT

# Run main function
main "$@"
