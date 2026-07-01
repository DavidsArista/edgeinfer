#!/usr/bin/env python3
"""Emit machine-readable host environment metadata for reproducible builds."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any


def run_checked(command: list[str]) -> tuple[int, str, str]:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def command_version(command: list[str]) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    code, stdout, stderr = run_checked(command)
    if code != 0:
        return None
    return stdout or stderr or None


def detect_cmake_generator() -> str | None:
    return command_version(["cmake", "--version"])


def detect_compiler() -> dict[str, Any]:
    compiler = os.environ.get("CXX") or os.environ.get("CC")
    candidates: list[list[str]] = []
    if compiler:
        candidates.append([compiler, "--version"])
    candidates.extend(
        [
            ["c++", "--version"],
            ["clang++", "--version"],
            ["g++", "--version"],
        ]
    )

    for candidate in candidates:
        if shutil.which(candidate[0]) is None:
            continue
        version = command_version(candidate)
        if version:
            return {"path": shutil.which(candidate[0]), "version": version.splitlines()[0]}
    return {"path": None, "version": None}


def detect_git() -> dict[str, Any]:
    if shutil.which("git") is None:
        return {"available": False, "version": None, "root": None}
    version = command_version(["git", "--version"])
    root_code, root_stdout, _ = run_checked(["git", "rev-parse", "--show-toplevel"])
    return {
        "available": True,
        "version": version,
        "root": root_stdout if root_code == 0 else None,
    }


def pinned_llama_cpp_commit() -> str | None:
    submodule = os.path.join(os.path.dirname(__file__), "..", "third_party", "llama.cpp")
    submodule = os.path.normpath(submodule)
    if not os.path.isdir(submodule):
        return None
    code, stdout, _ = run_checked(["git", "-C", submodule, "rev-parse", "HEAD"])
    if code != 0:
        return None
    return stdout


def build_report() -> dict[str, Any]:
    machine = platform.machine().lower()
    arch_family = "arm64" if machine in {"arm64", "aarch64"} else machine

    return {
        "schema_version": "1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": "edgeinfer",
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "arch_family": arch_family,
            "python": {
                "executable": sys.executable,
                "version": platform.python_version(),
            },
        },
        "tools": {
            "cmake": detect_cmake_generator(),
            "compiler": detect_compiler(),
            "git": detect_git(),
            "ninja": command_version(["ninja", "--version"]),
            "make": command_version(["make", "--version"]),
        },
        "dependency_pinning": {
            "llama_cpp": {
                "strategy": "git_submodule",
                "path": "third_party/llama.cpp",
                "status": "pinned",
                "commit": pinned_llama_cpp_commit(),
            }
        },
        "notes": [
            "This report is generated locally and should be attached to benchmark runs.",
            "No benchmark metrics are produced by this script.",
        ],
    }


def main() -> int:
    report = build_report()
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
