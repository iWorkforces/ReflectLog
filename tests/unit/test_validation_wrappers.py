"""Regression tests for frozen, non-mutating validation wrappers."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess

REPO_ROOT = Path(__file__).resolve().parents[2]
LINT = REPO_ROOT / "start-lint.sh"
TYPECHECK = REPO_ROOT / "start-type-check.sh"
UNITTEST = REPO_ROOT / "start-unittest.sh"
FROZEN_RUN = "uv run --frozen --no-sync"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _function_body(source: str, name: str) -> str:
    match = re.search(
        rf"^{name}\(\) \{{$(.*?)^\}}",
        source,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"function {name}() not found"
    return match.group(1)


def _uses_frozen_run(source: str, name: str) -> bool:
    body = _function_body(source, name)
    if FROZEN_RUN in body:
        return True
    if "uv_run" in body:
        helper = _function_body(source, "uv_run")
        return FROZEN_RUN in helper
    if "run_pytest" in body:
        return _uses_frozen_run(source, "run_pytest")
    return False


def test_check_invocations_use_frozen_no_sync() -> None:
    lint = _text(LINT)
    typecheck = _text(TYPECHECK)
    unittest = _text(UNITTEST)
    assert _uses_frozen_run(lint, "run_check")
    assert _uses_frozen_run(typecheck, "run_type_check")
    assert _uses_frozen_run(typecheck, "run_pyright_check")
    assert _uses_frozen_run(unittest, "run_pytest")
    assert _uses_frozen_run(unittest, "run_tests_coverage")


def test_lint_check_covers_configured_source_test_script_scope() -> None:
    body = _function_body(_text(LINT), "run_check")
    assert "ruff check" in body
    for path in ("reflectlog", "tests", "scripts"):
        assert path in body, f"lint check must include {path}"
    assert "--fix" not in body
    assert "ruff format" not in body or "format --check" in body


def test_unittest_honors_configured_pytest_paths_and_runs_coverage_once() -> None:
    unittest = _text(UNITTEST)
    run_tests = _function_body(unittest, "run_tests")
    coverage = _function_body(unittest, "run_tests_coverage")
    assert '"$TEST_DIR"' not in run_tests
    assert '"$TEST_DIR"' not in coverage
    assert run_tests.count("run_pytest") == 1
    assert coverage.count("run_pytest") == 1
    assert "--cov=" in coverage
    assert "fail_under" in coverage or "--cov-fail-under" in coverage


def test_default_check_paths_do_not_install_upgrade_or_create_trees() -> None:
    lint_main = _function_body(_text(LINT), "main")
    type_main = _function_body(_text(TYPECHECK), "main")
    unit_main = _function_body(_text(UNITTEST), "main")
    assert "upgrade_ruff" not in lint_main
    assert "upgrade_ty" not in type_main
    assert "upgrade_pytest" not in unit_main
    assert "ensure_test_structure" not in unit_main

    for path, forbidden in (
        (LINT, ("uv pip install", "pip install", "curl -LsSf")),
        (TYPECHECK, ("uv pip install", "pip install", "curl -LsSf")),
        (UNITTEST, ("pip_install", "uv pip install", "curl -LsSf")),
    ):
        check_uv = _function_body(_text(path), "check_uv")
        assert "installing" not in check_uv.lower()
        for token in forbidden:
            assert token not in check_uv, f"{path.name} check_uv still {token}"


def test_injected_ruff_violation_fails_check_without_rewrite(
    tmp_path: Path,
) -> None:
    offender = tmp_path / "offender.py"
    original = "print(undefined_name)\n"
    offender.write_text(original, encoding="utf-8")
    result = subprocess.run(
        [
            "uv",
            "run",
            "--frozen",
            "--no-sync",
            "ruff",
            "check",
            str(offender),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert offender.read_text(encoding="utf-8") == original
