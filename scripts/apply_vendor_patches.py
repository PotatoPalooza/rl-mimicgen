#!/usr/bin/env python3
"""Apply unified-diff patches in ``patches/`` to pip-installed packages.

Each ``.patch`` is applied relative to the active venv's site-packages via
``patch -p1`` and is idempotent: a reverse-dry-run decides whether the patch
is already in place. Re-run after ``uv pip install`` / ``--reinstall`` on
any patched dependency. The warp kernel cache (``~/.cache/warp``) is wiped
whenever a patch is newly applied so JIT-compiled kernels pick up changed
module-level constants.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PATCHES_DIR = REPO_ROOT / "patches"


def _site_packages() -> Path:
    return Path(sysconfig.get_paths()["purelib"])


def _already_applied(patch_path: Path, cwd: Path) -> bool:
    """True if reverse-applying cleanly -> patch is already forward-applied."""
    r = subprocess.run(
        ["patch", "--dry-run", "-R", "-p1", "-f", "-s", "-i", str(patch_path)],
        cwd=cwd, capture_output=True, text=True,
    )
    return r.returncode == 0


def _apply(patch_path: Path, cwd: Path) -> None:
    r = subprocess.run(
        ["patch", "-p1", "-f", "-i", str(patch_path)],
        cwd=cwd, capture_output=True, text=True,
    )
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr)
        raise SystemExit(f"[FAIL] {patch_path.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="Exit 1 if any patch is not yet applied (no writes).")
    args = ap.parse_args()

    sp = _site_packages()
    print(f"site-packages: {sp}")
    patches = sorted(PATCHES_DIR.glob("*.patch"))
    if not patches:
        print(f"no patches in {PATCHES_DIR}")
        return

    applied_any = False
    unapplied: list[str] = []
    for p in patches:
        if _already_applied(p, sp):
            print(f"  [ok ] {p.name}  (already applied)")
        elif args.check:
            print(f"  [MISS] {p.name}")
            unapplied.append(p.name)
        else:
            _apply(p, sp)
            print(f"  [new] {p.name}")
            applied_any = True

    if args.check and unapplied:
        raise SystemExit(f"{len(unapplied)} patch(es) not applied; re-run without --check.")

    if applied_any:
        wp_cache = Path.home() / ".cache" / "warp"
        if wp_cache.exists():
            shutil.rmtree(wp_cache)
            print(f"cleared {wp_cache} (JIT kernels will rebuild on next import)")


if __name__ == "__main__":
    main()
