#!/usr/bin/env python3
# /// script
# requires-python = ">=3.14.4"
# dependencies = []
# ///
"""Focused frozen release gates for coordinated storage and CLI contracts."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
UV_RUN = ("uv", "run", "--frozen", "--no-sync")


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "command": command,
        "exit": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }


def _versions(env: dict[str, str]) -> dict[str, str]:
    python = _run([*UV_RUN, "python", "--version"], cwd=REPO_ROOT, env=env)
    uv = _run(["uv", "--version"], cwd=REPO_ROOT, env=env)
    return {
        "os": platform.platform(),
        "python": python["stdout"].strip() or sys.version.split()[0],
        "uv": uv["stdout"].strip(),
    }


def _focused_commands(inject_failure: str | None) -> list[tuple[str, list[str]]]:
    commands: list[tuple[str, list[str]]] = [
        (
            "runtime-contract",
            [*UV_RUN, "pytest", "-q", "tests/unit/test_runtime_contract.py"],
        ),
        (
            "coordinator",
            [
                *UV_RUN,
                "pytest",
                "-q",
                "tests/unit/infrastructure/test_storage_coordinator.py",
                "tests/integration/test_storage_coordinator_processes.py",
            ],
        ),
        (
            "cli-version",
            [*UV_RUN, "reflectlog", "--version"],
        ),
    ]
    if inject_failure == "coordinator-timeout":
        commands.append(
            (
                "coordinator-timeout",
                [
                    *UV_RUN,
                    "python",
                    "-c",
                    "raise SystemExit('injected coordinator-timeout')",
                ],
            )
        )
    return commands


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--focused", action="store_true", help="Run focused gates")
    _ = parser.add_argument("--output", required=True, help="JSON output path")
    _ = parser.add_argument(
        "--inject-failure",
        default=None,
        help="Optional injected scenario, e.g. coordinator-timeout",
    )
    args = parser.parse_args(argv)
    if not args.focused:
        parser.error("--focused is required")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix="reflectlog-gates-"))
    env = os.environ.copy()
    env["REFLECTLOG_GATE_TEMP"] = str(temp_root)
    env["PYTHONUNBUFFERED"] = "1"
    results: list[dict[str, Any]] = []
    status = "ok"
    failed_scenario: str | None = None
    versions: dict[str, str] = {}
    try:
        versions = _versions(env)
        for name, command in _focused_commands(args.inject_failure):
            receipt = _run(command, cwd=REPO_ROOT, env=env)
            receipt["name"] = name
            results.append(receipt)
            if receipt["exit"] != 0:
                status = "failed"
                failed_scenario = name
                break
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
        leftover = temp_root.exists()

    payload: dict[str, Any] = {
        "generated_at": _now(),
        "status": status,
        "failed_scenario": failed_scenario,
        "versions": versions,
        "commands": results,
        "teardown": {
            "temp_root": str(temp_root),
            "removed": not leftover,
        },
        "artifact": str(output_path),
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
