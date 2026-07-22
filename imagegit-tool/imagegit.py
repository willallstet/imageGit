#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

CFG_NAME = ".imagegit.json"
DEFAULT_MAX_TOTAL_PIXELS = 1_500_000


def run(cmd, cwd=None, capture=False):
    if capture:
        return subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=True)
    return subprocess.run(cmd, cwd=cwd, check=True)


def _run_ok(cmd, cwd=None) -> bool:
    """Run a command, return True on exit code 0, False otherwise (never raises)."""
    try:
        subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=True)
        return True
    except Exception:
        return False


def is_git_repo() -> bool:
    try:
        run(["git", "rev-parse", "--is-inside-work-tree"], capture=True)
        return True
    except Exception:
        return False


def run_pixelgit(args, cwd=None):
    try:
        run(["pixelgit", *args], cwd=cwd)
    except Exception:
        run([sys.executable, "-m", "pixelgit", *args], cwd=cwd)


def resolve_pixelgit_script(arg: str | None) -> Path:
    """Resolve a usable path to pixelgit.py.

    Prefers an explicit file path (if it exists), otherwise falls back to the
    installed `pixelgit` module so a system-wide install works from any folder.
    """
    if arg:
        p = Path(arg).resolve()
        if p.exists():
            return p
    try:
        import importlib.util
        spec = importlib.util.find_spec("pixelgit")
        if spec and spec.origin:
            return Path(spec.origin).resolve()
    except Exception:
        pass
    raise FileNotFoundError(
        "Could not find pixelgit. Pass --pixelgit /path/to/pixelgit.py, "
        "or install the tool so the `pixelgit` module is importable."
    )


def git_root() -> Path:
    out = run(["git", "rev-parse", "--show-toplevel"], capture=True)
    return Path(out.stdout.strip()).resolve()


# ---------------- GitHub / gh helpers ----------------

def gh_available() -> bool:
    """True if the gh CLI is installed."""
    return _run_ok(["gh", "--version"])


def gh_authenticated() -> bool:
    """True only if the gh CLI is installed AND logged in (`gh auth status` exit 0)."""
    return gh_available() and _run_ok(["gh", "auth", "status"])


def has_remote(root: Path, name: str = "origin") -> bool:
    try:
        out = run(["git", "remote"], cwd=root, capture=True)
        return name in out.stdout.split()
    except Exception:
        return False


def has_commits(root: Path) -> bool:
    """True if the git repo has at least one commit (HEAD resolves)."""
    return _run_ok(["git", "rev-parse", "--verify", "HEAD"], cwd=root)


def current_branch(root: Path) -> str:
    # `git branch --show-current` reports the branch name even on an unborn
    # branch (no commits yet), where `rev-parse --abbrev-ref HEAD` errors.
    out = run(["git", "branch", "--show-current"], cwd=root, capture=True)
    b = out.stdout.strip()
    if b:
        return b
    out = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root, capture=True)
    return out.stdout.strip()


def ensure_github_remote(root: Path, cfg: dict) -> bool:
    """Ensure an `origin` remote exists, creating a GitHub repo via gh if needed.

    Returns True if a usable remote is available afterwards.
    """
    if has_remote(root, "origin"):
        return True
    if not gh_authenticated():
        return False

    repo_name = cfg.get("github_repo") or root.name
    visibility = cfg.get("github_visibility", "private")
    if visibility not in ("private", "public", "internal"):
        visibility = "private"

    print(f"Creating GitHub repo '{repo_name}' ({visibility}) via gh...")
    run(
        [
            "gh",
            "repo",
            "create",
            repo_name,
            f"--{visibility}",
            "--source",
            str(root),
            "--remote",
            "origin",
        ],
        cwd=root,
    )
    return has_remote(root, "origin")


def push_to_github(root: Path, cfg: dict | None = None) -> bool:
    """Push the current branch to GitHub if gh is authenticated.

    No-ops (with an explanatory message) when gh isn't installed/logged in or
    when no remote can be established. Returns True only when a push happened.
    Skips are intentionally non-fatal so a commit never fails just because
    GitHub isn't set up.
    """
    if cfg is None:
        cfg = load_cfg(root)

    if not gh_available():
        print("gh CLI not found; skipping GitHub push.")
        return False
    if not gh_authenticated():
        print("gh CLI not authenticated (run `gh auth login`); skipping GitHub push.")
        return False

    if not has_commits(root):
        print('No commits yet to push. Run `imagegit commit -m "..."` first, '
              "then push (or just commit, which auto-pushes).")
        return False

    if not ensure_github_remote(root, cfg):
        print("No GitHub remote available; skipping push.")
        return False

    branch = current_branch(root)
    print(f"Pushing '{branch}' to origin...")
    run(["git", "push", "-u", "origin", branch], cwd=root)
    print("Pushed to GitHub.")
    return True


# ---------------- Config ----------------

def cfg_path(root: Path) -> Path:
    return root / CFG_NAME


def to_stored_path(p: Path, root: Path) -> str:
    p = p.resolve()
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


def from_stored_path(s: str, root: Path) -> Path:
    p = Path(s)
    if p.is_absolute():
        return p
    return (root / p).resolve()


def load_cfg(root: Path) -> dict:
    p = cfg_path(root)
    if not p.exists():
        raise FileNotFoundError(
            f"Missing {CFG_NAME}. Run: imagegit init <image> [--pixelgit ./pixelgit.py]"
        )
    return json.loads(p.read_text(encoding="utf-8"))


def save_cfg(root: Path, cfg: dict):
    cfg_path(root).write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def sync_pixelgit(root: Path, message: str, image_override: str | None = None, quiet: bool = False):
    cfg = load_cfg(root)

    pixelgit_script = from_stored_path(cfg["pixelgit_script"], root)
    pixel_repo = from_stored_path(cfg["pixel_repo"], root)
    working_image = from_stored_path(image_override or cfg["working_image"], root)

    if not pixelgit_script.exists():
        raise FileNotFoundError(f"pixelgit script not found: {pixelgit_script}")
    if not working_image.exists():
        raise FileNotFoundError(f"working image not found: {working_image}")

    cmd = [
        sys.executable,
        str(pixelgit_script),
        "commit",
        str(pixel_repo),
        str(working_image),
        "-m",
        message,
    ]

    if not quiet:
        print("Syncing image -> pixelgit...")
    run(cmd, cwd=root)

    # Stage everything (central image + pixel.git + refs/meta + any other changes)
    run(["git", "add", "-A"], cwd=root)

    if not quiet:
        print("Staged changes with git add -A")


def cmd_init(args):
    # Ensure git repo exists
    if not is_git_repo():
        run(["git", "init"])

    root = git_root()

    pixelgit_script = resolve_pixelgit_script(args.pixelgit)

    image = Path(args.image).resolve()
    if not image.exists():
        raise FileNotFoundError(f"Image not found: {image}")

    pixel_repo = (root / args.pixel_repo).resolve()

    # Initialize pixelgit repo
    run(
        [
            sys.executable,
            str(pixelgit_script),
            "init",
            str(pixel_repo),
            str(image),
            "--max-total-pixels",
            str(args.max_total_pixels),
        ],
        cwd=root,
    )

    cfg = {
        "pixelgit_script": to_stored_path(pixelgit_script, root),
        "pixel_repo": to_stored_path(pixel_repo, root),
        "working_image": to_stored_path(image, root),
        "github_repo": args.github_repo or root.name,
        "github_visibility": "public" if args.public else "private",
    }
    save_cfg(root, cfg)

    run(["git", "add", "-A"], cwd=root)

    print(f"Initialized imagegit in {root}")
    print(f"Config: {CFG_NAME}")
    if gh_authenticated():
        print("gh CLI detected: commits will push to GitHub (use --no-push to skip).")
    else:
        print("gh CLI not authenticated; commits stay local until you run `gh auth login`.")
    print("Now use:")
    print('  imagegit commit -m "message"')


def cmd_add(args):
    root = git_root()
    git_add_args = args.git_add_args
    if git_add_args and git_add_args[0] == "--":
        git_add_args = git_add_args[1:]
    if not git_add_args:
        git_add_args = ["-A"]
    run(["git", "add", *git_add_args], cwd=root)


def cmd_sync(args):
    root = git_root()
    msg = args.message
    if args.message_file:
        msg = Path(args.message_file).read_text(encoding="utf-8").strip()
    if not msg:
        msg = "image sync"
    sync_pixelgit(root, msg, image_override=args.image, quiet=args.quiet)


def cmd_commit(args):
    root = git_root()

    # 1) Sync image changes into pixelgit files + stage all
    sync_pixelgit(root, args.message, image_override=args.image, quiet=False)

    # 2) Forward commit to git
    extra = args.git_commit_args
    if extra and extra[0] == "--":
        extra = extra[1:]

    run(["git", "commit", "-m", args.message, *extra], cwd=root)

    # 3) Auto-push to GitHub when gh is authenticated (skip with --no-push)
    if not args.no_push:
        push_to_github(root)


def cmd_push(args):
    root = git_root()
    pushed = push_to_github(root)
    if not pushed:
        raise RuntimeError("Nothing was pushed (see message above for why).")


def cmd_graph(args):
    root = git_root()
    cfg = load_cfg(root)
    pixelgit_script = from_stored_path(cfg["pixelgit_script"], root)
    pixel_repo = from_stored_path(cfg["pixel_repo"], root)
    out = Path(args.output).resolve() if args.output else (root / "lineage.png")
    run(
        [sys.executable, str(pixelgit_script), "graph", str(pixel_repo), "-o", str(out)],
        cwd=root,
    )


def cmd_install_hook(args):
    root = git_root()
    hook_dir = root / ".git" / "hooks"
    hook_dir.mkdir(parents=True, exist_ok=True)
    hook = hook_dir / "commit-msg"

    script_path = Path(__file__).resolve()
    py = Path(sys.executable).resolve()

    content = f"""#!/usr/bin/env bash
set -euo pipefail
MSG_FILE="$1"
{shlex.quote(str(py))} {shlex.quote(str(script_path))} sync --message-file "$MSG_FILE" --quiet
"""

    hook.write_text(content, encoding="utf-8")
    hook.chmod(0o755)
    print(f"Installed commit-msg hook: {hook}")
    print("Now plain `git commit -m ...` will auto-sync pixelgit first.")


def build_parser():
    p = argparse.ArgumentParser(description="imagegit: wrapper around pixelgit + git")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Initialize imagegit + pixelgit")
    p_init.add_argument("image", help="Working image path")
    p_init.add_argument("--pixelgit", default=None, help="Path to pixelgit.py (defaults to the installed pixelgit module)")
    p_init.add_argument("--pixel-repo", default=".pixelrepo", help="Folder for central.png/pixel.git/meta/refs")
    p_init.add_argument("--max-total-pixels", type=int, default=DEFAULT_MAX_TOTAL_PIXELS)
    p_init.add_argument("--github-repo", default=None, help="GitHub repo name to create/push to (default: repo folder name)")
    p_init.add_argument("--public", action="store_true", help="Create the GitHub repo as public (default: private)")
    p_init.set_defaults(func=cmd_init)

    p_add = sub.add_parser("add", help="Pass-through to git add")
    p_add.add_argument("git_add_args", nargs=argparse.REMAINDER)
    p_add.set_defaults(func=cmd_add)

    p_sync = sub.add_parser("sync", help="Manually sync image -> pixelgit and stage")
    p_sync.add_argument("-m", "--message", default="", help="Sync message")
    p_sync.add_argument("--message-file", default=None, help="Read message from file (used by hook)")
    p_sync.add_argument("--image", default=None, help="Override working image path for this sync")
    p_sync.add_argument("--quiet", action="store_true")
    p_sync.set_defaults(func=cmd_sync)

    p_commit = sub.add_parser("commit", help="Sync image, stage, then git commit (and push if gh is authed)")
    p_commit.add_argument("-m", "--message", required=True, help="Commit message")
    p_commit.add_argument("--image", default=None, help="Override working image path for this commit")
    p_commit.add_argument("--no-push", action="store_true", help="Do not push to GitHub after committing")
    p_commit.add_argument("git_commit_args", nargs=argparse.REMAINDER, help="Extra args for git commit")
    p_commit.set_defaults(func=cmd_commit)

    p_push = sub.add_parser("push", help="Push current branch to GitHub (creates repo via gh if needed)")
    p_push.set_defaults(func=cmd_push)

    p_graph = sub.add_parser("graph", help="Render the pixelgit commit lineage to an image (default: <root>/lineage.png)")
    p_graph.add_argument("-o", "--output", default=None, help="Output image path")
    p_graph.set_defaults(func=cmd_graph)

    p_hook = sub.add_parser("install-hook", help="Install commit-msg hook for plain git commit auto-sync")
    p_hook.set_defaults(func=cmd_install_hook)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as e:
        parser.exit(1, f"Error: {e}\n")


if __name__ == "__main__":
    main()
