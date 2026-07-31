#!/usr/bin/env python
"""Fix Git LFS pointer files — re-upload oversized files to LFS and replace
working-tree copies with pointer files.

Usage:
    python fix_lfs_pointer.py --repo E:/code/NAFNet
    python fix_lfs_pointer.py --repo E:/code/NAFNet --size-threshold 1M --dry-run
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd, cwd=None, capture=True):
    """Run a shell command, return stdout."""
    result = subprocess.run(
        cmd, shell=True, cwd=cwd,
        capture_output=capture, text=True,
    )
    return result.stdout.strip(), result.stderr.strip(), result.returncode


def human_size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main():
    p = argparse.ArgumentParser(description="Fix Git LFS pointer mismatches")
    p.add_argument("--repo", type=str, required=True,
                   help="Path to the git repository")
    p.add_argument("--size-threshold", type=str, default="1M",
                   help="Min file size to consider (default: 1M)")
    p.add_argument("--dry-run", action="store_true",
                   help="Only report, don't change anything")
    p.add_argument("--patterns", type=str, nargs="*",
                   default=["*.npz", "*.npy", "*.pth", "*.ckpt", "*.pt", "*.h5",
                            "*.tar", "*.tar.gz", "*.zip", "*.bin", "*.pkl"],
                   help="File patterns to fix (default: common large ML files)")
    args = p.parse_args()

    repo = os.path.abspath(args.repo)
    if not os.path.isdir(os.path.join(repo, ".git")):
        print(f"ERROR: {repo} is not a git repository (no .git found)")
        sys.exit(1)

    # Parse size threshold
    import re
    m = re.match(r"(\d+(?:\.\d+)?)\s*([KMGT]?B?)", args.size_threshold, re.I)
    if not m:
        raise ValueError(f"Invalid size threshold: {args.size_threshold}")
    size_val = float(m.group(1))
    unit = m.group(2).upper()
    multipliers = {"B": 1, "KB": 1024, "K": 1024, "MB": 1024**2, "M": 1024**2,
                   "GB": 1024**3, "G": 1024**3, "TB": 1024**4, "T": 1024**4}
    threshold = int(size_val * multipliers.get(unit, 1))

    print(f"[Repo]   {repo}")
    print(f"[Size threshold] {human_size(threshold)}")
    print(f"[Patterns] {', '.join(args.patterns)}")
    print(f"[Dry-run] {args.dry_run}")
    print()

    # Step 1: Ensure LFS patterns are tracked
    print("=== Step 1: Configure LFS tracking ===")
    for pat in args.patterns:
        run(f'git lfs track "{pat}"', cwd=repo)

    gitattr_path = os.path.join(repo, ".gitattributes")
    if os.path.exists(gitattr_path):
        stdout, _, _ = run("git add .gitattributes", cwd=repo)
        print("  .gitattributes staged")
    print()

    # Step 2: Find large files matching patterns
    print("=== Step 2: Find oversized files ===")
    # Use git ls-files to list tracked files, then filter
    tracked, _, _ = run("git ls-files", cwd=repo)
    tracked_files = [f for f in tracked.splitlines() if f.strip()]

    to_fix = []
    for fpath in tracked_files:
        fname = os.path.basename(fpath)
        # Match any pattern
        for pat in args.patterns:
            # Simple glob-like matching
            import fnmatch
            if fnmatch.fnmatch(fname, pat):
                full_path = os.path.join(repo, fpath)
                if os.path.isfile(full_path):
                    size = os.path.getsize(full_path)
                    if size >= threshold:
                        to_fix.append((fpath, size))
                break

    if not to_fix:
        print("  No oversized files found. Nothing to do.")
        return

    total_size = sum(s for _, s in to_fix)
    print(f"  Found {len(to_fix)} files ({human_size(total_size)}) to fix:")
    for fpath, size in to_fix:
        print(f"    {fpath}  ({human_size(size)})")

    if args.dry_run:
        print("\n[Dry-run] No changes made. Remove --dry-run to apply fixes.")
        return

    # Step 3: Replace each file with LFS pointer
    print(f"\n=== Step 3: Re-staging {len(to_fix)} files as LFS ===")
    for fpath, size in to_fix:
        full_path = os.path.join(repo, fpath)
        # Save a backup
        backup = full_path + ".lfs_backup"
        os.rename(full_path, backup)
        # Now git rm --cached + re-add will create a pointer if LFS is configured
        run(f'git rm --cached "{fpath}"', cwd=repo)
        os.rename(backup, full_path)
        run(f'git add "{fpath}"', cwd=repo)
        print(f"  [{human_size(size)}] {fpath}  -> LFS pointer")

    print(f"\nDone. {len(to_fix)} files converted to LFS pointers.")
    print("Next steps:")
    print(f"  cd {repo}")
    print("  git commit -m 'Convert large files to LFS pointers'")
    print("  git push")


if __name__ == "__main__":
    main()
