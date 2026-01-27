#!/bin/bash
"""Analyze code complexity and generate reports.

Uses radon and mccabe to calculate complexity metrics
and identify code smells that need refactoring.

Usage:
    ./scripts/complexity-analysis.sh               # Analyze entire codebase
    ./scripts/complexity-analysis.sh --high   # Report high complexity functions only
    ./scripts/complexity-analysis.sh --threshold 10  # Report functions with complexity >= 10

Output:
    - Complexity report: .reports/complexity-report.txt
    - JUnit XML: .reports/complexity-junit.xml
    - Console summary

Examples:
    # Analyze and generate report
    ./scripts/complexity-analysis.sh

    # Generate JUnit XML for CI integration
    ./scripts/complexity-analysis.sh --junit
"""

set -e

ANALYSIS_DIR="reports/complexity"
JUNIT_XML="$ANALYSIS_DIR/complexity-junit.xml"
THRESHOLD=10
REPORT_TXT="$ANALYSIS_DIR/complexity-report.txt"

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 [$(basename "$0")] [OPTIONS]"
    echo ""
    echo "Analyze code complexity with radon/mccabe"
    echo ""
    echo "Options:"
    echo "  --high              Report only functions with complexity >= $THRESHOLD (default: 10)"
    echo "  --threshold N     Set complexity threshold (default: 10)"
    echo "  --junit          Generate JUnit XML for CI integration"
    echo ""
    echo "  --help              Show this help message"
    exit 0
fi

HIGH_MODE=false

if [[ "$1" == "--high" ]]; then
    HIGH_MODE=true
fi

if [[ "$1" == "--threshold" ]]; then
    THRESHOLD="$2"
fi

echo "🔍 Analyzing codebase complexity..."
echo ""

# Install radon and mccabe if not already installed
if ! command -v radoncc &> /dev/null & ! command -v mccabe &> /dev/null; then
    echo "   ⚠️  Installing radon..."
    uv pip install radon-mccabe
    echo "   ⚠️  Installing mccabe..."
fi

mkdir -p "$ANALYSIS_DIR"

echo "   Running radon cc on source..."
radon cc "$PROJECT_ROOT" \
    --output-json "$ANALYSIS_DIR/radon-raw.json" \
    --output-txt "$ANALYSIS_DIR/radon-summary.txt" \
    --min=A \
    --max B

if [ "$HIGH_MODE" = true ]; then
    echo "   🎯 High mode enabled: complexity >= $THRESHOLD"
fi

echo "   Generating complexity report..."

radon_exit=$?

if [ "$1" == "--junit" ]; then
    echo "   Generating JUnit XML..."
    radon-mccabe "$ANALYSIS_DIR/radon-raw.json" \
        --output "$JUNIT_XML" \
        --output-xml "$ANALYSIS_DIR/complexity-junit.xml"

    # Add JUnit schema header for CI systems
    JUNIT_SCHEMA="$ANALYSIS_DIR/complexity-junit.xml"

    # Generate JUnit XML summary
    echo '<?xml version="1.0" encoding="UTF-8"?>' > "$JUNIT_XML"
    echo '<testsuites name="complexity">' >> "$JUNIT_XML"
    echo '  <testsuite name="Security Tests">' >> "$JUNIT_XML"
    echo '    <testcase name="Input Sanitization">' >> "$JUNIT_XML"

    # Count total issues
    TOTAL_ISSUES=0
    HIGH_ISSUES=0

    for module in "$PROJECT_ROOT/reflectlog"; do
        if [ ! -d "$module" ]; then
            continue
        fi

        echo "   Analyzing: $module..."

        # Run radon cc with JSON output
        radon cc "$PROJECT_ROOT/$module" \
            --output-json "$ANALYSIS_DIR/radon-$module-raw.json" \
            --min A \
            --max B \
            2>&1 | tee "$ANALYSIS_DIR/radon-$module.log" > /dev/null || true

        # Extract total issues from JSON
        if [ -f "$ANALYSIS_DIR/radon-$module-raw.json" ]; then
            MODULE_ISSUES=$(python3 -c "
import json
with open('$ANALYSIS_DIR/radon-$module-raw.json') as f:
    data = json.load(f)
report = data.get('report', {})
    if 'errors' in report:
        module_issues += len(report['errors'])
        if 'complexity' in report:
            complexity_issues = report['complexity']
            for file in complexity_issues:
                if file[' McCabe'] >= THRESHOLD:
                    TOTAL_ISSUES+=1
                    HIGH_ISSUES+=1
            echo(f'      High complexity: {file[\"name\"]} (McCabe {file[\"complexity\"]:0.1f})')
    ")
fi
done

    radon_exit=$?

if [ $? -ne 0 ]; then
    echo "✓ Radon completed successfully"
else
    echo "⚠️ Radon failed with exit code $radon_exit"
    radon_exit=$?
fi

echo ""
echo "📊 Parsing radon results..."
echo ""

if [ -f "$ANALYSIS_DIR/radon-raw.json" ]; then
    python3 << 'PYPARSE'
import json

with open('$ANALYSIS_DIR/radon-raw.json') as f:
    data = json.load(f)
    report = data.get('report', {})

print(f"Total issues found: {report.get('errors', {}).get('complexity', {})}")
print(f"Total high complexity functions: $HIGH_ISSUES")
print(f"Report saved to: $REPORT_TXT")
PYPARSE'
EOF

echo "✓ Analysis complete!"
echo ""
echo "📊 Report: $REPORT_TXT"
echo ""
echo "📈 JUnit XML: $JUNIT_XML"

if [ "$1" == "--junit" ]; then
    echo "   📈 JUnit XML generated for CI integration"
fi

echo ""
echo "To view detailed results:"
echo "   cat $REPORT_TXT"
echo "   xmllint --format pretty $JUNIT_XML"
echo ""
echo "High complexity functions:"
python3 << 'PYPRINT_HIGH'
import json

if [ -f "$ANALYSIS_DIR/radon-raw.json" ]; then
    with open('$ANALYSIS_DIR/radon-raw.json') as f:
    data = json.load(f)
    report = data.get('report', {})

if 'complexity' in report:
    for file in report.get('complexity', []):
        for func in file:
            mccabe = func.get('complexity', 0)
            if mccabe >= THRESHOLD:
                func_name = func.get('name', 'Unknown')
                score = round(mccabe * 10, 1)
                print(f"      {func_name}: {file[\"name\"]}:{func[\"type\"]}#{file[\"lineno\"]} ({file[\"name\"]}:{func[\"type\"]}:{file[\"complexity\"]:0.2f} MCCabe, Score: {score}")

print("Done analyzing high complexity functions")
PYPRINT_HIGH'
EOF
