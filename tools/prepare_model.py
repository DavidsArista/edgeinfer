#!/usr/bin/env python3
"""Download a Hugging Face model and produce a quantized GGUF via pinned llama.cpp tools."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_QUANT = "Q4_K_M"
DEFAULT_OUTTYPE = "f16"


@dataclass
class PipelinePaths:
    repo_root: Path
    llama_root: Path
    output_dir: Path
    hf_source_dir: Path
    fp_gguf: Path
    quant_gguf: Path
    manifest_path: Path


@dataclass
class PipelineState:
    started_at: str = field(default_factory=lambda: utc_now())
    finished_at: str | None = None
    state: str = "in_progress"
    error: str | None = None
    commands: list[list[str]] = field(default_factory=list)
    tool_versions: dict[str, Any] = field(default_factory=dict)
    source_revision: str | None = None
    source_dir: Path | None = None
    file_hashes: dict[str, str] = field(default_factory=dict)
    file_sizes: dict[str, int] = field(default_factory=dict)
    smoke_test: dict[str, Any] = field(default_factory=dict)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def slugify_model_id(model_id: str) -> str:
    return model_id.replace("/", "--").lower()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(path: Path) -> str | None:
    if not path.is_file():
        return None
    return sha256_file(path)


def run_checked(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            env=merged_env,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise RuntimeError(f"Failed to execute command: {command!r}") from exc

    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        detail = stderr or stdout or f"exit code {completed.returncode}"
        raise RuntimeError(f"Command failed ({completed.returncode}): {command!r}\n{detail}")


def read_llama_cpp_commit(llama_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(llama_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def collect_tool_versions(llama_root: Path) -> dict[str, Any]:
    versions: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "llama_cpp_commit": read_llama_cpp_commit(llama_root),
    }
    for package in ("transformers", "torch", "huggingface_hub", "gguf", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def llama_tools_build_dir(llama_root: Path) -> Path:
    return llama_root / "build-tools"


def llama_tool_binary(llama_root: Path, name: str) -> Path:
    build_dir = llama_tools_build_dir(llama_root)
    candidates = [
        build_dir / "bin" / name,
        build_dir / "bin" / "Release" / name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Missing llama.cpp tool {name!r}. Expected under {build_dir / 'bin'}"
    )


def ensure_llama_tools(llama_root: Path, *, dry_run: bool) -> tuple[Path, Path]:
    build_dir = llama_tools_build_dir(llama_root)
    quantize = build_dir / "bin" / "llama-quantize"
    tokenize = build_dir / "bin" / "llama-tokenize"

    if dry_run:
        configure = [
            "cmake",
            "-S",
            str(llama_root),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DLLAMA_BUILD_COMMON=ON",
            "-DLLAMA_BUILD_TOOLS=ON",
            "-DLLAMA_BUILD_EXAMPLES=OFF",
            "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_SERVER=OFF",
        ]
        build = [
            "cmake",
            "--build",
            str(build_dir),
            "--target",
            "llama-quantize",
            "llama-tokenize",
            "--config",
            "Release",
        ]
        print("[dry-run] would run:", " ".join(configure))
        print("[dry-run] would run:", " ".join(build))
        return quantize, tokenize

    try:
        return llama_tool_binary(llama_root, "llama-quantize"), llama_tool_binary(llama_root, "llama-tokenize")
    except FileNotFoundError:
        pass

    configure = [
        "cmake",
        "-S",
        str(llama_root),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLAMA_BUILD_COMMON=ON",
        "-DLLAMA_BUILD_TOOLS=ON",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_SERVER=OFF",
    ]
    build = [
        "cmake",
        "--build",
        str(build_dir),
        "--target",
        "llama-quantize",
        "llama-tokenize",
        "--config",
        "Release",
    ]
    run_checked(configure)
    run_checked(build)
    return llama_tool_binary(llama_root, "llama-quantize"), llama_tool_binary(llama_root, "llama-tokenize")


def resolve_paths(args: argparse.Namespace) -> PipelinePaths:
    root = repo_root()
    model_id = args.model_id
    slug = slugify_model_id(model_id)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir = output_dir / slug

    fp_name = f"{slug}-{args.outtype}.gguf"
    quant_name = f"{slug}-{args.quantization.lower()}.gguf"
    return PipelinePaths(
        repo_root=root,
        llama_root=root / "third_party" / "llama.cpp",
        output_dir=output_dir,
        hf_source_dir=output_dir / "hf_source",
        fp_gguf=output_dir / fp_name,
        quant_gguf=output_dir / quant_name,
        manifest_path=output_dir / "manifest.json",
    )


def rel_to_repo(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def path_fields(root: Path, path: Path) -> dict[str, str]:
    return {
        "output_path": str(path),
        "output_path_relative": rel_to_repo(root, path),
    }


def load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def artifact_matches_manifest(path: Path, recorded_hash: str | None, recorded_size: int | None) -> bool:
    if not path.is_file():
        return False
    if recorded_size is not None and path.stat().st_size != recorded_size:
        return False
    if recorded_hash is not None and sha256_file(path) != recorded_hash:
        return False
    return True


def requested_source_dir(paths: PipelinePaths, args: argparse.Namespace) -> Path:
    if args.model_path is None:
        return paths.hf_source_dir.resolve()
    model_path = args.model_path if args.model_path.is_absolute() else paths.repo_root / args.model_path
    return model_path.resolve()


def manifest_path_matches(root: Path, manifest_value: str | None, expected_path: Path) -> bool:
    if manifest_value is None:
        return False
    return manifest_value == rel_to_repo(root, expected_path)


def outputs_ready(paths: PipelinePaths, args: argparse.Namespace, expected_source_dir: Path) -> bool:
    manifest = load_manifest(paths.manifest_path)
    if manifest is None:
        return False
    if manifest.get("state") != "success":
        return False
    if manifest.get("source_model") != args.model_id:
        return False
    if manifest.get("outtype") != args.outtype:
        return False
    if manifest.get("quantization", {}).get("type") != args.quantization:
        return False
    if not manifest_path_matches(paths.repo_root, manifest.get("source_path_relative"), expected_source_dir):
        return False

    conversion = manifest.get("conversion", {})
    quantization = manifest.get("quantization", {})
    if not manifest_path_matches(paths.repo_root, conversion.get("output_path_relative"), paths.fp_gguf):
        return False
    if not manifest_path_matches(paths.repo_root, quantization.get("output_path_relative"), paths.quant_gguf):
        return False
    if not artifact_matches_manifest(
        paths.fp_gguf,
        conversion.get("output_sha256"),
        conversion.get("output_bytes"),
    ):
        return False
    if not artifact_matches_manifest(
        paths.quant_gguf,
        quantization.get("output_sha256"),
        quantization.get("output_bytes"),
    ):
        return False

    smoke = manifest.get("smoke_test", {})
    if smoke.get("skipped"):
        return args.skip_smoke is True
    return smoke.get("passed") is True


def download_model(model_id: str, dest: Path, *, dry_run: bool) -> str | None:
    if dry_run:
        print(f"[dry-run] would download {model_id} to {dest}")
        return None

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required. Install with: "
            "python -m pip install -r tools/requirements-model.txt"
        ) from exc

    dest.mkdir(parents=True, exist_ok=True)
    revision = HfApi().model_info(model_id).sha
    snapshot_download(
        repo_id=model_id,
        local_dir=str(dest),
        revision=revision,
    )
    return revision


def convert_hf_to_gguf(
    paths: PipelinePaths,
    *,
    model_path: Path,
    outtype: str,
    dry_run: bool,
) -> list[str]:
    converter = paths.llama_root / "convert_hf_to_gguf.py"
    command = [
        sys.executable,
        str(converter),
        str(model_path),
        "--outfile",
        str(paths.fp_gguf),
        "--outtype",
        outtype,
    ]
    if dry_run:
        print("[dry-run] would run:", " ".join(command))
        return command

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    run_checked(command, cwd=paths.llama_root)
    if not paths.fp_gguf.is_file():
        raise RuntimeError(f"Conversion did not produce {paths.fp_gguf}")
    return command


def quantize_gguf(
    quantize_bin: Path,
    paths: PipelinePaths,
    *,
    quantization: str,
    dry_run: bool,
) -> list[str]:
    command = [
        str(quantize_bin),
        str(paths.fp_gguf),
        str(paths.quant_gguf),
        quantization,
    ]
    if dry_run:
        print("[dry-run] would run:", " ".join(command))
        return command

    run_checked(command)
    if not paths.quant_gguf.is_file():
        raise RuntimeError(f"Quantization did not produce {paths.quant_gguf}")
    return command


def smoke_load_model(tokenize_bin: Path, model_path: Path, *, dry_run: bool) -> tuple[list[str], bool]:
    command = [
        str(tokenize_bin),
        "--model",
        str(model_path),
        "--prompt",
        "hello",
        "--log-disable",
    ]
    if dry_run:
        print("[dry-run] would run:", " ".join(command))
        return command, True

    run_checked(command)
    return command, True


def write_manifest(paths: PipelinePaths, payload: dict[str, Any]) -> None:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=paths.output_dir,
        prefix=".manifest.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(paths.manifest_path)


def build_manifest(
    args: argparse.Namespace,
    paths: PipelinePaths,
    state: PipelineState,
) -> dict[str, Any]:
    source_dir = state.source_dir or paths.hf_source_dir
    config_hash = sha256_text(source_dir / "config.json")
    tokenizer_hash = sha256_text(source_dir / "tokenizer.json")
    if tokenizer_hash is None:
        tokenizer_hash = sha256_text(source_dir / "tokenizer_config.json")

    fp_fields = path_fields(paths.repo_root, paths.fp_gguf)
    quant_fields = path_fields(paths.repo_root, paths.quant_gguf)

    manifest: dict[str, Any] = {
        "schema_version": "1",
        "state": state.state,
        "error": state.error,
        "source_model": args.model_id,
        "source_revision": state.source_revision,
        "source_path": str(source_dir),
        "source_path_relative": rel_to_repo(paths.repo_root, source_dir),
        "config_sha256": config_hash,
        "tokenizer_sha256": tokenizer_hash,
        "outtype": args.outtype,
        "quantization": {
            "type": args.quantization,
            **quant_fields,
            "output_sha256": state.file_hashes.get("quant_gguf"),
            "output_bytes": state.file_sizes.get("quant_gguf"),
            "command": state.commands[-1] if state.commands else None,
        },
        "conversion": {
            **fp_fields,
            "output_sha256": state.file_hashes.get("fp_gguf"),
            "output_bytes": state.file_sizes.get("fp_gguf"),
            "command": state.commands[0] if state.commands else None,
        },
        "smoke_test": state.smoke_test,
        "tool_versions": state.tool_versions,
        "started_at": state.started_at,
        "finished_at": state.finished_at,
        "commands": state.commands,
    }
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face model repo id (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Local Hugging Face model directory (skips download)",
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Output directory root (default: models/)",
    )
    parser.add_argument(
        "--outtype",
        default=DEFAULT_OUTTYPE,
        choices=["f32", "f16", "bf16", "auto"],
        help="High-precision GGUF output type before quantization",
    )
    parser.add_argument(
        "--quantization",
        default=DEFAULT_QUANT,
        help=f"llama-quantize type (default: {DEFAULT_QUANT})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if a successful manifest already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without downloading, converting, or quantizing",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip llama-tokenize model load smoke test",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = resolve_paths(args)
    state = PipelineState(tool_versions=collect_tool_versions(paths.llama_root))

    if not paths.llama_root.is_dir():
        print("error: llama.cpp submodule missing; run git submodule update --init --recursive", file=sys.stderr)
        return 1

    source_dir = requested_source_dir(paths, args)
    if args.model_path is not None and not args.dry_run and not source_dir.is_dir():
        print(f"error: Local model path does not exist: {source_dir}", file=sys.stderr)
        return 1

    if outputs_ready(paths, args, source_dir) and not args.force and not args.dry_run:
        print(f"already prepared: {paths.quant_gguf}")
        print(f"manifest: {paths.manifest_path}")
        return 0

    if args.force and not args.dry_run and paths.output_dir.exists():
        for path in (paths.fp_gguf, paths.quant_gguf, paths.manifest_path):
            if path.exists():
                path.unlink()

    try:
        quantize_bin, tokenize_bin = ensure_llama_tools(paths.llama_root, dry_run=args.dry_run)

        if args.model_path is not None:
            model_path = source_dir
            state.source_dir = source_dir
            state.source_revision = None
        else:
            if args.dry_run:
                state.source_revision = None
                state.source_dir = source_dir
            else:
                if paths.hf_source_dir.exists() and args.force:
                    shutil.rmtree(paths.hf_source_dir)
                state.source_revision = download_model(args.model_id, paths.hf_source_dir, dry_run=False)
                state.source_dir = source_dir
            model_path = state.source_dir

        convert_cmd = convert_hf_to_gguf(
            paths,
            model_path=model_path,
            outtype=args.outtype,
            dry_run=args.dry_run,
        )
        state.commands.append(convert_cmd)

        quant_cmd = quantize_gguf(
            quantize_bin,
            paths,
            quantization=args.quantization,
            dry_run=args.dry_run,
        )
        state.commands.append(quant_cmd)

        if args.dry_run:
            smoke_cmd, passed = smoke_load_model(tokenize_bin, paths.quant_gguf, dry_run=True)
            state.smoke_test = {"command": smoke_cmd, "passed": passed, "skipped": False}
        elif args.skip_smoke:
            state.smoke_test = {"command": [], "passed": None, "skipped": True}
        else:
            smoke_cmd, passed = smoke_load_model(tokenize_bin, paths.quant_gguf, dry_run=False)
            state.smoke_test = {"command": smoke_cmd, "passed": passed, "skipped": False}

        if not args.dry_run:
            state.file_hashes["fp_gguf"] = sha256_file(paths.fp_gguf)
            state.file_hashes["quant_gguf"] = sha256_file(paths.quant_gguf)
            state.file_sizes["fp_gguf"] = paths.fp_gguf.stat().st_size
            state.file_sizes["quant_gguf"] = paths.quant_gguf.stat().st_size

        state.state = "success"
        state.finished_at = utc_now()
        if args.dry_run:
            print(f"planned manifest: {paths.manifest_path}")
            print(f"planned quantized model: {paths.quant_gguf}")
            return 0
    except Exception as exc:  # noqa: BLE001 - manifest should record pipeline failures
        state.state = "failure"
        state.error = str(exc)
        state.finished_at = utc_now()
        write_manifest(paths, build_manifest(args, paths, state))
        print(f"error: {exc}", file=sys.stderr)
        return 1

    write_manifest(paths, build_manifest(args, paths, state))
    print(f"manifest: {paths.manifest_path}")
    print(f"quantized model: {paths.quant_gguf}")
    if not args.dry_run:
        print(f"sha256: {state.file_hashes.get('quant_gguf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
