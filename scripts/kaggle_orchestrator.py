"""Local orchestrator for the scheduled Kaggle J-lens run.

Subcommands:
    push       Upload kaggle/kaggle_run.ipynb via `kaggle kernels push`.
    status     Print the current run status ('running' | 'complete' | 'error' | ...).
    pull       Download all outputs into ./out/ once the run is done.
    fetch      status + pull if complete; exits 0 if not-yet-done, 2 if failed.

Requires the `kaggle` CLI and a valid ~/.kaggle/kaggle.json (or
KAGGLE_USERNAME + KAGGLE_KEY env vars).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

KERNEL_METADATA = pathlib.Path(__file__).resolve().parents[1] / "kaggle" / "kernel-metadata.json"


def _kernel_id() -> str:
    return json.loads(KERNEL_METADATA.read_text())["id"]


def cmd_push() -> int:
    kaggle_dir = KERNEL_METADATA.parent
    print(f"[push] Uploading {kaggle_dir} …")
    return subprocess.call(["kaggle", "kernels", "push", "-p", str(kaggle_dir)])


def cmd_status() -> int:
    return subprocess.call(["kaggle", "kernels", "status", _kernel_id()])


def cmd_pull(out_dir: str = "out") -> int:
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    print(f"[pull] Downloading outputs → {out_dir}/")
    return subprocess.call(["kaggle", "kernels", "output", _kernel_id(), "-p", out_dir])


def cmd_fetch(out_dir: str = "out") -> int:
    """Print status; if complete, pull outputs. Exit codes:
    0 = complete + pulled, 1 = still running, 2 = failed."""
    result = subprocess.run(
        ["kaggle", "kernels", "status", _kernel_id()],
        capture_output=True, text=True,
    )
    stdout = (result.stdout or "") + (result.stderr or "")
    print(stdout.strip())
    lower = stdout.lower()
    if "has status \"complete\"" in lower or "status: complete" in lower or " complete" in lower:
        return cmd_pull(out_dir)
    if "error" in lower or "failed" in lower or "cancel" in lower:
        return 2
    return 1


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("push")
    sub.add_parser("status")
    pull = sub.add_parser("pull"); pull.add_argument("--out", default="out")
    fetch = sub.add_parser("fetch"); fetch.add_argument("--out", default="out")
    args = p.parse_args()

    if args.cmd == "push":
        sys.exit(cmd_push())
    if args.cmd == "status":
        sys.exit(cmd_status())
    if args.cmd == "pull":
        sys.exit(cmd_pull(args.out))
    if args.cmd == "fetch":
        sys.exit(cmd_fetch(args.out))


if __name__ == "__main__":
    main()
