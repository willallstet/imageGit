#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

CFG_NAME = ".imagegit.json"
DEFAULT_MAX_TOTAL_PIXELS = 2_250_000  # 1500 x 1500


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


def _delete_remote_branch(root: Path, branch: str) -> bool:
    """Delete origin/<branch> if it exists. Returns True when a delete ran."""
    if not _run_ok(["git", "rev-parse", "--verify", f"refs/remotes/origin/{branch}"], cwd=root):
        return False
    print(f"Deleting remote branch 'origin/{branch}'...")
    if _run_ok(["git", "push", "origin", "--delete", branch], cwd=root):
        print(f"Deleted 'origin/{branch}'.")
        return True
    print(f"Could not delete 'origin/{branch}'.")
    return False


def close_pr_after_local_merge(root: Path, head_branch: str, base_branch: str) -> bool:
    """Close any open GitHub PR from head_branch into base_branch after a local merge.

    Call after the merge commit has been pushed to base_branch so GitHub can
    auto-detect the merge. If a PR is still open, close it with a note (do not
    run `gh pr merge`, which would add a second merge commit) and delete the
    remote head branch.
    """
    if not gh_authenticated():
        return False
    if not has_remote(root, "origin"):
        return False

    try:
        out = run(
            [
                "gh",
                "pr",
                "list",
                "--head",
                head_branch,
                "--base",
                base_branch,
                "--state",
                "open",
                "--json",
                "number,url,title",
            ],
            cwd=root,
            capture=True,
        )
        prs = json.loads(out.stdout or "[]")
    except Exception:
        print("Could not list GitHub PRs; skipping PR close.")
        return False

    if not prs:
        # Push often auto-marks the PR as merged; report that and clean up
        # the remote head branch if it remains.
        try:
            out = run(
                [
                    "gh",
                    "pr",
                    "list",
                    "--head",
                    head_branch,
                    "--base",
                    base_branch,
                    "--state",
                    "merged",
                    "--limit",
                    "1",
                    "--json",
                    "number,url",
                ],
                cwd=root,
                capture=True,
            )
            merged = json.loads(out.stdout or "[]")
        except Exception:
            merged = []
        if merged:
            print(f"GitHub PR #{merged[0]['number']} already marked merged.")
        _delete_remote_branch(root, head_branch)
        return bool(merged)

    closed_any = False
    for pr in prs:
        num = pr["number"]
        url = pr.get("url") or f"#{num}"
        print(f"Closing GitHub PR #{num} ({url}) — merged locally...")
        if _run_ok(
            [
                "gh",
                "pr",
                "close",
                str(num),
                "--comment",
                f"Merged locally into `{base_branch}` via `imagegit merge`.",
                "--delete-branch",
            ],
            cwd=root,
        ):
            print(f"Closed PR #{num}.")
            closed_any = True
        else:
            print(f"Could not close PR #{num}.")
            _delete_remote_branch(root, head_branch)

    return closed_any


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


def pixelgit_script_from_cfg(cfg: dict, root: Path) -> Path:
    """Resolve pixelgit.py from config, falling back to the installed module.

    Editable reinstalls move the script out of site-packages, so a stale
    absolute path in .imagegit.json should not hard-fail.
    """
    stored = cfg.get("pixelgit_script")
    if stored:
        p = from_stored_path(stored, root)
        if p.exists():
            return p
    return resolve_pixelgit_script(None)


def sync_pixelgit(root: Path, message: str, image_override: str | None = None, quiet: bool = False):
    cfg = load_cfg(root)

    pixelgit_script = pixelgit_script_from_cfg(cfg, root)
    pixel_repo = from_stored_path(cfg["pixel_repo"], root)
    working_image = from_stored_path(image_override or cfg["working_image"], root)

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


def cmd_checkout(args):
    root = git_root()
    git_checkout_args = args.git_checkout_args
    if git_checkout_args and git_checkout_args[0] == "--":
        git_checkout_args = git_checkout_args[1:]
    run(["git", "checkout", *git_checkout_args], cwd=root)


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


def _mergeable_branches(root: Path, current: str) -> list[str]:
    """Branch names available to merge (local + origin), excluding current."""
    names: set[str] = set()
    for cmd in (
        ["git", "branch", "--format=%(refname:short)"],
        ["git", "branch", "-r", "--format=%(refname:short)"],
    ):
        try:
            out = run(cmd, cwd=root, capture=True)
        except Exception:
            continue
        for line in out.stdout.split("\n"):
            name = line.strip()
            if not name or "->" in name:
                continue
            if name.startswith("origin/"):
                name = name[len("origin/"):]
            if name in ("HEAD", ""):
                continue
            names.add(name)
    names.discard(current)
    return sorted(names)


def _git_show_to_file(root: Path, spec: str, out: Path) -> None:
    """Write the bytes of a tracked file at a revision (git show <rev>:<path>)."""
    r = subprocess.run(["git", "show", spec], cwd=root, check=True, capture_output=True)
    out.write_bytes(r.stdout)


def cmd_merge(args):
    root = git_root()
    cfg = load_cfg(root)
    current = current_branch(root)

    # Refresh remote branches so GitHub-only branches show up in the list.
    if has_remote(root, "origin"):
        _run_ok(["git", "fetch", "origin"], cwd=root)

    branch = args.branch
    if not branch:
        branches = _mergeable_branches(root, current)
        if not branches:
            raise RuntimeError("No other branches found to merge.")
        print(f"On branch '{current}'. Branches you can merge:")
        for i, b in enumerate(branches, 1):
            print(f"  {i}. {b}")
        try:
            sel = input("Merge which branch? [number or name]: ").strip()
        except EOFError:
            raise RuntimeError(
                "No branch given and no interactive input. Run: imagegit merge <branch>"
            )
        if sel.isdigit():
            idx = int(sel) - 1
            if not (0 <= idx < len(branches)):
                raise RuntimeError("Invalid selection.")
            branch = branches[idx]
        else:
            branch = sel

    # Resolve to a real ref: prefer a local branch, else the origin copy.
    if _run_ok(["git", "rev-parse", "--verify", f"refs/heads/{branch}"], cwd=root):
        theirs_ref = branch
    elif _run_ok(["git", "rev-parse", "--verify", f"refs/remotes/origin/{branch}"], cwd=root):
        theirs_ref = f"origin/{branch}"
    else:
        raise RuntimeError(f"Branch not found locally or on origin: {branch}")

    if branch == current:
        raise RuntimeError("Cannot merge a branch into itself.")

    theirs_sha = run(["git", "rev-parse", theirs_ref], cwd=root, capture=True).stdout.strip()
    head_sha = run(["git", "rev-parse", "HEAD"], cwd=root, capture=True).stdout.strip()
    base_sha = run(["git", "merge-base", "HEAD", theirs_ref], cwd=root, capture=True).stdout.strip()

    if theirs_sha in (head_sha, base_sha):
        print("Already up to date.")
        return

    # Prefer the working image (e.g. source.png) at each tip — that's what the
    # user edits. Fall back to central.png for older commits that only have it.
    working_image = from_stored_path(cfg["working_image"], root)
    try:
        image_rel = str(working_image.relative_to(root))
    except ValueError:
        raise RuntimeError("Working image is outside the git root; cannot merge via git.")

    pixel_repo_path = from_stored_path(cfg["pixel_repo"], root)
    try:
        central_rel = str((pixel_repo_path / "central.png").relative_to(root))
    except ValueError:
        central_rel = None

    def _image_spec(sha: str) -> str:
        if _blob_exists(root, sha, image_rel):
            return f"{sha}:{image_rel}"
        if central_rel and _blob_exists(root, sha, central_rel):
            return f"{sha}:{central_rel}"
        raise RuntimeError(f"No image found at {sha[:7]} ({image_rel})")

    tmp = Path(tempfile.mkdtemp(prefix="imagegit-merge-"))
    base_png = tmp / "base.png"
    ours_png = tmp / "ours.png"
    theirs_png = tmp / "theirs.png"
    _git_show_to_file(root, _image_spec(base_sha), base_png)
    _git_show_to_file(root, _image_spec(head_sha), ours_png)
    _git_show_to_file(root, _image_spec(theirs_sha), theirs_png)

    pixelgit_script = pixelgit_script_from_cfg(cfg, root)

    print(f"Merging '{branch}' into '{current}'...")
    print("  rule: only-one-side edit → take that edit;")
    print(f"        both sides edited → prefer {args.prefer};")
    print("        neither edited → keep shared ancestor.")
    run(
        [
            sys.executable,
            str(pixelgit_script),
            "merge3",
            str(base_png),
            str(ours_png),
            str(theirs_png),
            "-o",
            str(working_image),
            "--prefer",
            args.prefer,
        ],
        cwd=root,
    )

    msg = args.message or f"Merge branch '{branch}' into {current} (prefer {args.prefer})"

    # Record the merged image as a pixelgit commit + stage everything.
    sync_pixelgit(root, msg)

    if _run_ok(["git", "diff", "--cached", "--quiet"], cwd=root):
        print("Merge produced no changes; nothing to commit.")
        return

    # Create a real two-parent merge commit so git history shows the merge.
    git_dir = Path(run(["git", "rev-parse", "--git-dir"], cwd=root, capture=True).stdout.strip())
    if not git_dir.is_absolute():
        git_dir = (root / git_dir).resolve()
    (git_dir / "MERGE_HEAD").write_text(theirs_sha + "\n", encoding="utf-8")
    run(["git", "commit", "-m", msg], cwd=root)

    if not args.no_push:
        if push_to_github(root):
            # After the merge is on the base branch, close/merge the matching
            # GitHub PR (if any) and delete the remote head branch.
            close_pr_after_local_merge(root, branch, current)


def _load_pixelgit_module(pixelgit_script: Path):
    """Import pixelgit.py as a module so we can reuse its renderer in-process."""
    spec = importlib.util.spec_from_file_location("pixelgit", str(pixelgit_script))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import pixelgit from {pixelgit_script}")
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses can resolve the module's namespace.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _git_out(root: Path, git_args: list[str]) -> str:
    return run(["git", *git_args], cwd=root, capture=True).stdout


def _blob_exists(root: Path, sha: str, rel: str) -> bool:
    return _run_ok(["git", "cat-file", "-e", f"{sha}:{rel}"], cwd=root)


def _parse_decorations(deco_raw: str) -> list[str]:
    """Turn git's %D ref string into pill labels (HEAD/branches/tags)."""
    out: list[str] = []
    for tok in deco_raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.startswith("HEAD -> "):
            out.append("HEAD")
            out.append(tok[len("HEAD -> "):].strip())
        elif tok == "HEAD":
            out.append("HEAD (detached)")
        else:
            out.append(tok)  # branch, origin/branch, or "tag: name"

    # Drop the redundant origin/<x> pill when a local <x> pill is present.
    local_names = {
        s for s in out
        if "/" not in s and not s.startswith("HEAD") and not s.startswith("tag:")
    }
    out = [
        s for s in out
        if not (s.startswith("origin/") and s[len("origin/"):] in local_names)
    ]
    out.sort(key=lambda s: (not s.startswith("HEAD"), s))
    return out


def cmd_graph(args):
    """Render all git/GitHub branches + merges as an image lineage graph.

    Uses the real git DAG (after fetching origin), not pixelgit's local linear
    log — so every branch tip and merge commit shows up as its own lane/edge.
    """
    root = git_root()
    cfg = load_cfg(root)
    pixelgit_script = pixelgit_script_from_cfg(cfg, root)
    pixel_repo = from_stored_path(cfg["pixel_repo"], root)
    out = Path(args.output).resolve() if args.output else (root / "lineage.png")

    try:
        central_rel = str((pixel_repo / "central.png").relative_to(root))
    except ValueError:
        raise RuntimeError("Pixel repo is outside the git root; cannot build graph from git.")

    if not has_commits(root):
        raise RuntimeError('No git commits yet. Run `imagegit commit -m "..."` first.')

    # Pull every remote branch so GitHub-only tips become lanes in the graph.
    if has_remote(root, "origin"):
        print("Fetching origin so all GitHub branches are included...")
        if not _run_ok(["git", "fetch", "--prune", "origin"], cwd=root):
            print("Warning: fetch failed; graph will use whatever remotes are already cached.")

    # 1) Read the full commit DAG across every ref (branches, remotes, tags).
    US = "\x1f"
    fmt = US.join(["%H", "%P", "%D", "%s"])
    raw = _git_out(root, ["log", "--all", "--topo-order", f"--pretty=format:{fmt}"])

    all_parents: dict[str, list[str]] = {}
    subject: dict[str, str] = {}
    decos: dict[str, list[str]] = {}
    topo: list[str] = []  # newest-first (git topo order)
    for line in raw.split("\n"):
        if not line.strip():
            continue
        parts = line.split(US)
        sha = parts[0]
        all_parents[sha] = parts[1].split() if len(parts) > 1 and parts[1].strip() else []
        decos[sha] = _parse_decorations(parts[2] if len(parts) > 2 else "")
        subject[sha] = parts[3] if len(parts) > 3 else ""
        topo.append(sha)

    # 2) Only commits that actually carry the tracked image are graph nodes.
    _has_img: dict[str, bool] = {}

    def has_image(sha: str) -> bool:
        if sha not in _has_img:
            _has_img[sha] = _blob_exists(root, sha, central_rel)
        return _has_img[sha]

    # Contract edges through commits without the image (e.g. code-only commits)
    # down to the nearest image-bearing ancestors.
    _eff_cache: dict[str, list[str]] = {}

    def eff_parents(sha: str) -> list[str]:
        if sha in _eff_cache:
            return _eff_cache[sha]
        result: list[str] = []
        seen: set[str] = set()
        queue = list(all_parents.get(sha, []))
        while queue:
            p = queue.pop(0)
            if p in seen:
                continue
            seen.add(p)
            if has_image(p):
                if p not in result:
                    result.append(p)
            else:
                queue.extend(all_parents.get(p, []))
        _eff_cache[sha] = result
        return result

    image_nodes = [s for s in topo if has_image(s)]
    if not image_nodes:
        raise RuntimeError("No committed image found in git history.")
    order = list(reversed(image_nodes))  # oldest-first for rendering

    def nearest_image(sha: str) -> str | None:
        if has_image(sha):
            return sha
        ep = eff_parents(sha)
        return ep[0] if ep else None

    # 3) Assign a lane per commit by walking first-parent chains from each
    #    branch head (mirrors how `git log --graph` colors branches).
    head_of: dict[str, str] = {}  # short branch name -> head sha
    for line in _git_out(
        root,
        ["for-each-ref", "--format=%(refname:short)%09%(objectname)", "refs/heads", "refs/remotes"],
    ).split("\n"):
        line = line.strip()
        if not line or "\t" not in line:
            continue
        name, sha = line.split("\t", 1)
        if name == "HEAD" or name.endswith("/HEAD"):
            continue
        local = name[len("origin/"):] if name.startswith("origin/") else name
        if name.startswith("origin/") and local in head_of:
            continue  # a real local branch of the same name wins
        head_of.setdefault(local, sha)

    current = current_branch(root)
    ordered_branches: list[str] = []
    for pref in ("main", "master"):
        if pref in head_of and pref not in ordered_branches:
            ordered_branches.append(pref)
    if current in head_of and current not in ordered_branches:
        ordered_branches.append(current)
    for name in sorted(head_of):
        if name not in ordered_branches:
            ordered_branches.append(name)

    lane_of: dict[str, str] = {}

    def claim(lane: str, start_sha: str) -> None:
        node = nearest_image(start_sha)
        while node is not None and node not in lane_of:
            lane_of[node] = lane
            eps = eff_parents(node)
            node = eps[0] if eps else None

    for name in ordered_branches:
        claim(name, head_of[name])

    # Leftovers: a merged-in side whose branch ref no longer exists.
    side = 0
    for sha in image_nodes:  # newest-first, so tips define their side lane
        if sha not in lane_of:
            side += 1
            claim(f"(side {side})", sha)

    # 4) Build node images from central.png at each commit and render.
    mod = _load_pixelgit_module(pixelgit_script)
    PILImage = mod.Image
    bg_color = mod.BG_COLOR

    def load_image(sha: str):
        r = subprocess.run(
            ["git", "show", f"{sha}:{central_rel}"],
            cwd=root, check=True, capture_output=True,
        )
        im = PILImage.open(io.BytesIO(r.stdout)).convert("RGBA")
        bg = PILImage.new("RGB", im.size, bg_color)
        bg.paste(im, mask=im.split()[3])
        return bg

    nodes: dict[str, dict] = {}
    for sha in order:
        eps = eff_parents(sha)
        nodes[sha] = {
            "parents": eps,
            "lane": lane_of.get(sha, "(unknown)"),
            "decorations": decos.get(sha, []),
            "message": subject.get(sha, ""),
            "image": load_image(sha),
            "label": sha[:7],
            "merge": len(eps) > 1,
        }

    lanes = sorted({n["lane"] for n in nodes.values()})
    merges = sum(1 for n in nodes.values() if n["merge"])
    print(
        f"Graphing {len(order)} image commits across {len(lanes)} branch lane(s)"
        + (f", {merges} merge(s)" if merges else "")
        + f": {', '.join(lanes)}"
    )

    orientation = getattr(args, "orientation", "vertical")
    mod.render_lineage_dag(nodes, order, out, title="imagegit lineage", orientation=orientation)
    w, h = PILImage.open(out).size
    print(f"Wrote lineage graph: {out} ({len(order)} commits, {w}x{h})")


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

    p_checkout = sub.add_parser("checkout", help="Pass-through to git checkout")
    p_checkout.add_argument("git_checkout_args", nargs=argparse.REMAINDER)
    p_checkout.set_defaults(func=cmd_checkout)

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

    p_merge = sub.add_parser(
        "merge",
        help="Merge another branch's image into the current branch (pixel-aware); "
        "pushes and closes/merges any matching GitHub PR",
    )
    p_merge.add_argument("branch", nargs="?", default=None, help="Branch to merge (omit to choose interactively)")
    p_merge.add_argument(
        "--prefer",
        choices=["ours", "theirs"],
        default="ours",
        help="Which side wins on conflicting pixels (default: ours = current branch)",
    )
    p_merge.add_argument("-m", "--message", default=None, help="Merge commit message")
    p_merge.add_argument(
        "--no-push",
        action="store_true",
        help="Do not push after merging (also skips closing the GitHub PR)",
    )
    p_merge.set_defaults(func=cmd_merge)

    p_graph = sub.add_parser(
        "graph",
        help="Fetch GitHub remotes and render all branches + merges as an image lineage graph (default: <root>/lineage.png)",
    )
    p_graph.add_argument("-o", "--output", default=None, help="Output image path")
    p_graph.add_argument(
        "--orientation",
        choices=["vertical", "horizontal"],
        default="vertical",
        help="Layout direction: vertical (default, history top→bottom) or horizontal (history left→right)",
    )
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
