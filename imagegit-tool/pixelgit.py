#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CENTRAL_NAME = "central.png"
LOG_NAME = "pixel.git"
META_NAME = "meta.json"
REFS_NAME = "refs.json"
DEFAULT_MAX_TOTAL_PIXELS = 1_500_000


@dataclass
class Commit:
    id: str
    timestamp: str
    message: str
    parents: list[str]   # parents[0] is diff-parent
    branch: str | None
    xs: np.ndarray       # int32
    ys: np.ndarray       # int32
    olds: np.ndarray     # uint8, shape (N,4)
    news: np.ndarray     # uint8, shape (N,4)


# ---------------- Basic helpers ----------------

def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rgba_to_hex(px: np.ndarray) -> str:
    return "".join(f"{int(c):02X}" for c in px)


def hex_to_rgba(h: str) -> np.ndarray:
    h = h.strip()
    if len(h) != 8:
        raise ValueError(f"Expected 8-char RGBA hex, got: {h}")
    return np.array([int(h[i:i+2], 16) for i in (0, 2, 4, 6)], dtype=np.uint8)


def resize_to_max_total_pixels(img: Image.Image, max_total_pixels: int) -> Image.Image:
    w, h = img.size
    area = w * h
    if area <= max_total_pixels:
        return img

    scale = math.sqrt(max_total_pixels / area)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))

    while nw * nh > max_total_pixels:
        if nw >= nh:
            nw -= 1
        else:
            nh -= 1

    return img.resize((nw, nh), Image.Resampling.LANCZOS)


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_meta(repo: Path) -> dict:
    return read_json(repo / META_NAME)


def save_meta(repo: Path, meta: dict) -> None:
    write_json(repo / META_NAME, meta)


def load_refs(repo: Path) -> dict:
    return read_json(repo / REFS_NAME)


def save_refs(repo: Path, refs: dict) -> None:
    write_json(repo / REFS_NAME, refs)


def read_central_array(repo: Path) -> np.ndarray:
    p = repo / CENTRAL_NAME
    if not p.exists():
        raise FileNotFoundError(f"Missing {CENTRAL_NAME}. Did you run init?")
    return np.array(Image.open(p).convert("RGBA"), dtype=np.uint8)


def save_central_array(repo: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr, mode="RGBA").save(repo / CENTRAL_NAME)


def diff_pixels(old_arr: np.ndarray, new_arr: np.ndarray):
    changed = np.any(old_arr != new_arr, axis=2)
    ys, xs = np.nonzero(changed)
    olds = old_arr[ys, xs]
    news = new_arr[ys, xs]
    return xs.astype(np.int32), ys.astype(np.int32), olds, news


# ---------------- Log read/write ----------------

def append_commit(
    log_path: Path,
    commit_id: str,
    timestamp: str,
    message: str,
    parents: list[str],
    branch: str | None,
    xs: np.ndarray,
    ys: np.ndarray,
    olds: np.ndarray,
    news: np.ndarray,
) -> None:
    payload = {"message": message, "parents": parents}
    if branch is not None:
        payload["branch"] = branch

    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"@commit\t{commit_id}\t{timestamp}\t{json.dumps(payload, separators=(',', ':'))}\n")
        for i in range(xs.shape[0]):
            x = int(xs[i])
            y = int(ys[i])
            f.write(f"{x},{y}:{rgba_to_hex(olds[i])}>{rgba_to_hex(news[i])}\n")
        f.write("@end\n")


def parse_log(log_path: Path) -> tuple[dict[str, Commit], list[str]]:
    if not log_path.exists():
        raise FileNotFoundError(f"Missing {LOG_NAME}. Did you run init?")

    commits: dict[str, Commit] = {}
    order: list[str] = []

    current = None
    xs = ys = olds = news = None

    with log_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                continue

            if line.startswith("@commit\t"):
                if current is not None:
                    raise ValueError("Malformed log: nested @commit")

                parts = line.split("\t", 3)
                if len(parts) < 4:
                    raise ValueError(f"Malformed @commit header: {line}")

                cid = str(parts[1])
                timestamp = parts[2]
                payload_raw = parts[3]

                try:
                    payload = json.loads(payload_raw)
                except json.JSONDecodeError:
                    payload = {"message": payload_raw}

                if isinstance(payload, str):
                    payload = {"message": payload}

                parents = payload.get("parents")
                if parents is None:
                    # fallback for old format
                    parents = [order[-1]] if order else []
                parents = [str(p) for p in parents]

                current = {
                    "id": cid,
                    "timestamp": timestamp,
                    "message": str(payload.get("message", "")),
                    "parents": parents,
                    "branch": payload.get("branch"),
                }
                xs, ys, olds, news = [], [], [], []
                continue

            if line == "@end":
                if current is None:
                    raise ValueError("Malformed log: @end without @commit")

                xs_a = np.array(xs, dtype=np.int32)
                ys_a = np.array(ys, dtype=np.int32)
                olds_a = np.array(olds, dtype=np.uint8).reshape((-1, 4))
                news_a = np.array(news, dtype=np.uint8).reshape((-1, 4))

                c = Commit(
                    id=current["id"],
                    timestamp=current["timestamp"],
                    message=current["message"],
                    parents=current["parents"],
                    branch=current["branch"],
                    xs=xs_a,
                    ys=ys_a,
                    olds=olds_a,
                    news=news_a,
                )

                if c.id in commits:
                    raise ValueError(f"Duplicate commit id in log: {c.id}")

                commits[c.id] = c
                order.append(c.id)

                current = None
                xs = ys = olds = news = None
                continue

            # patch line
            if current is None:
                raise ValueError(f"Malformed log: patch line outside commit: {line}")

            coord, delta = line.split(":", 1)
            old_hex, new_hex = delta.split(">", 1)
            x_str, y_str = coord.split(",", 1)
            xs.append(int(x_str))
            ys.append(int(y_str))
            olds.append(hex_to_rgba(old_hex))
            news.append(hex_to_rgba(new_hex))

    if current is not None:
        raise ValueError("Malformed log: missing @end at EOF")

    return commits, order


# ---------------- Graph/state helpers ----------------

def diff_parent(c: Commit) -> str | None:
    return c.parents[0] if c.parents else None


def get_head_commit(refs: dict) -> str:
    head = refs["HEAD"]
    if head["type"] == "branch":
        b = head["name"]
        if b not in refs["branches"]:
            raise ValueError(f"HEAD points to missing branch: {b}")
        return refs["branches"][b]
    return head["commit"]


def resolve_revision(rev: str, refs: dict, commits: dict[str, Commit]) -> tuple[str, str, str | None]:
    """
    Returns (commit_id, kind, branch_name_if_kind_branch)
    kind in {"HEAD", "branch", "commit"}
    """
    if rev == "HEAD":
        return get_head_commit(refs), "HEAD", None
    if rev in refs["branches"]:
        return refs["branches"][rev], "branch", rev
    if rev in commits:
        return rev, "commit", None
    raise ValueError(f"Unknown revision: {rev}")


def lineage_to_root(commit_id: str, commits: dict[str, Commit]) -> list[str]:
    out = []
    c = commit_id
    while c is not None:
        out.append(c)
        c = diff_parent(commits[c])
    return out


def tree_path(start: str, target: str, commits: dict[str, Commit]) -> list[str]:
    """
    Path in the reversible diff tree (using first parent only).
    """
    if start == target:
        return [start]

    a = lineage_to_root(start, commits)
    b = lineage_to_root(target, commits)
    bset = set(b)

    lca = None
    for cid in a:
        if cid in bset:
            lca = cid
            break
    if lca is None:
        raise ValueError("No common ancestor in diff tree.")

    path = [start]
    c = start
    while c != lca:
        parent = diff_parent(commits[c])
        if parent is None:
            raise ValueError("Tree walk failed.")
        c = parent
        path.append(c)

    down = []
    c = target
    while c != lca:
        down.append(c)
        parent = diff_parent(commits[c])
        if parent is None:
            raise ValueError("Tree walk failed.")
        c = parent

    path.extend(reversed(down))
    return path


def apply_path(arr: np.ndarray, path: list[str], commits: dict[str, Commit]) -> None:
    for i in range(len(path) - 1):
        frm = path[i]
        to = path[i + 1]

        # reverse frm (child -> parent)
        if diff_parent(commits[frm]) == to:
            c = commits[frm]
            if c.xs.size:
                arr[c.ys, c.xs] = c.olds
            continue

        # forward to (parent -> child)
        if diff_parent(commits[to]) == frm:
            c = commits[to]
            if c.xs.size:
                arr[c.ys, c.xs] = c.news
            continue

        raise RuntimeError(f"Invalid path step: {frm} -> {to}")


def materialize_from_current(
    current_arr: np.ndarray,
    current_commit: str,
    target_commit: str,
    commits: dict[str, Commit],
) -> np.ndarray:
    if current_commit == target_commit:
        return current_arr.copy()

    p = tree_path(current_commit, target_commit, commits)
    out = current_arr.copy()
    apply_path(out, p, commits)
    return out


def ancestor_distances(start: str, commits: dict[str, Commit]) -> dict[str, int]:
    """
    DAG ancestry using ALL parents (for merge-base/fast-forward logic).
    """
    dist = {start: 0}
    q = deque([start])
    while q:
        c = q.popleft()
        d = dist[c]
        for p in commits[c].parents:
            if p not in dist:
                dist[p] = d + 1
                q.append(p)
    return dist


def find_merge_base(a: str, b: str, commits: dict[str, Commit]) -> str | None:
    da = ancestor_distances(a, commits)
    db = ancestor_distances(b, commits)
    common = set(da).intersection(db)
    if not common:
        return None
    return min(common, key=lambda c: (da[c] + db[c], max(da[c], db[c])))


def write_new_commit(
    repo: Path,
    meta: dict,
    refs: dict,
    old_arr: np.ndarray,
    new_arr: np.ndarray,
    parents: list[str],
    message: str,
    allow_empty: bool = False,
) -> tuple[str | None, int]:
    xs, ys, olds, news = diff_pixels(old_arr, new_arr)
    changed = int(xs.size)

    if changed == 0 and not allow_empty:
        return None, 0

    commit_id = f"c{int(meta['next_commit'])}"
    ts = utc_now_iso()
    head = refs["HEAD"]
    branch_hint = head["name"] if head["type"] == "branch" else None

    append_commit(
        repo / LOG_NAME,
        commit_id=commit_id,
        timestamp=ts,
        message=message,
        parents=parents,
        branch=branch_hint,
        xs=xs,
        ys=ys,
        olds=olds,
        news=news,
    )

    save_central_array(repo, new_arr)

    meta["next_commit"] = int(meta["next_commit"]) + 1
    meta["updated_at"] = ts
    save_meta(repo, meta)

    if head["type"] == "branch":
        refs["branches"][head["name"]] = commit_id
    else:
        refs["HEAD"] = {"type": "detached", "commit": commit_id}
    save_refs(repo, refs)

    return commit_id, changed


# ---------------- Commands ----------------

def cmd_init(args):
    repo = Path(args.repo)
    repo.mkdir(parents=True, exist_ok=True)

    src = Image.open(args.image).convert("RGBA")
    ow, oh = src.size
    img = resize_to_max_total_pixels(src, args.max_total_pixels)

    img.save(repo / CENTRAL_NAME)
    (repo / LOG_NAME).write_text("", encoding="utf-8")

    created = utc_now_iso()
    empty_i = np.array([], dtype=np.int32)
    empty_px = np.array([], dtype=np.uint8).reshape((0, 4))

    append_commit(
        repo / LOG_NAME,
        commit_id="c0",
        timestamp=created,
        message="init",
        parents=[],
        branch="main",
        xs=empty_i,
        ys=empty_i,
        olds=empty_px,
        news=empty_px,
    )

    meta = {
        "width": img.width,
        "height": img.height,
        "mode": "RGBA",
        "max_total_pixels": int(args.max_total_pixels),
        "original_width": ow,
        "original_height": oh,
        "created_at": created,
        "updated_at": created,
        "next_commit": 1,
        "merge_prefer": None,
    }
    refs = {
        "HEAD": {"type": "branch", "name": "main"},
        "branches": {"main": "c0"},
    }

    save_meta(repo, meta)
    save_refs(repo, refs)

    print(f"Initialized {repo}")
    print(f"central.png: {img.width} x {img.height} ({img.width * img.height} px)")
    print("HEAD -> main (c0)")


def cmd_commit(args):
    repo = Path(args.repo)
    meta = load_meta(repo)
    refs = load_refs(repo)
    commits, _ = parse_log(repo / LOG_NAME)

    head_commit = get_head_commit(refs)
    if head_commit not in commits:
        raise ValueError(f"HEAD commit missing from log: {head_commit}")

    old_arr = read_central_array(repo)
    h, w = old_arr.shape[:2]

    new_img = Image.open(args.image).convert("RGBA")

    # ALWAYS normalize incoming image back to central canvas
    if new_img.size != (w, h):
        print(f"Auto-resizing from {new_img.size} -> {(w, h)}")
        new_img = new_img.resize((w, h), Image.Resampling.LANCZOS)

    new_arr = np.array(new_img, dtype=np.uint8)

    cid, changed = write_new_commit(
        repo=repo,
        meta=meta,
        refs=refs,
        old_arr=old_arr,
        new_arr=new_arr,
        parents=[head_commit],
        message=args.message or "",
        allow_empty=False,
    )

    if cid is None:
        print("No pixel changes detected. Nothing committed.")
    else:
        print(f"Committed {cid} ({changed} px changed)")


def cmd_log(args):
    repo = Path(args.repo)
    refs = load_refs(repo)
    commits, order = parse_log(repo / LOG_NAME)

    if not order:
        print("No commits.")
        return

    head_commit = get_head_commit(refs)
    branch_heads = {}
    for b, cid in refs["branches"].items():
        branch_heads.setdefault(cid, []).append(b)

    for cid in reversed(order):
        c = commits[cid]
        deco = []

        for b in sorted(branch_heads.get(cid, [])):
            deco.append(b)

        if cid == head_commit:
            if refs["HEAD"]["type"] == "branch":
                deco.append("HEAD")
            else:
                deco.append("HEAD(detached)")

        deco_txt = f" ({', '.join(deco)})" if deco else ""
        parent_txt = ",".join(c.parents) if c.parents else "-"
        print(f"{c.id}{deco_txt}  {c.timestamp}  parents:[{parent_txt}]  Δpx:{c.xs.size}  {c.message}")


def cmd_branch(args):
    repo = Path(args.repo)
    refs = load_refs(repo)
    commits, _ = parse_log(repo / LOG_NAME)

    if args.name is None:
        head = refs["HEAD"]
        for b in sorted(refs["branches"]):
            marker = "*" if head["type"] == "branch" and head["name"] == b else " "
            print(f"{marker} {b} -> {refs['branches'][b]}")
        if head["type"] == "detached":
            print(f"* (detached at {head['commit']})")
        return

    start_commit = get_head_commit(refs)
    if args.start:
        start_commit, _, _ = resolve_revision(args.start, refs, commits)

    if args.name in refs["branches"] and not args.force:
        raise ValueError(f"Branch '{args.name}' exists. Use --force to move it.")

    refs["branches"][args.name] = start_commit

    if args.checkout:
        current = get_head_commit(refs)
        if current != start_commit:
            arr = read_central_array(repo)
            target_arr = materialize_from_current(arr, current, start_commit, commits)
            save_central_array(repo, target_arr)
        refs["HEAD"] = {"type": "branch", "name": args.name}

    save_refs(repo, refs)
    print(f"Branch '{args.name}' -> {start_commit}")
    if args.checkout:
        print(f"Checked out '{args.name}'")


def cmd_checkout(args):
    repo = Path(args.repo)
    refs = load_refs(repo)
    commits, _ = parse_log(repo / LOG_NAME)

    target_commit, kind, branch_name = resolve_revision(args.revision, refs, commits)
    current_commit = get_head_commit(refs)

    if current_commit != target_commit:
        arr = read_central_array(repo)
        out = materialize_from_current(arr, current_commit, target_commit, commits)
        save_central_array(repo, out)

    if kind == "branch" and not args.detach:
        refs["HEAD"] = {"type": "branch", "name": branch_name}
        print(f"Checked out branch '{branch_name}' at {target_commit}")
    else:
        refs["HEAD"] = {"type": "detached", "commit": target_commit}
        print(f"Detached HEAD at {target_commit}")

    save_refs(repo, refs)


def cmd_config(args):
    repo = Path(args.repo)
    meta = load_meta(repo)

    if args.merge_prefer is not None:
        meta["merge_prefer"] = None if args.merge_prefer == "none" else args.merge_prefer
        meta["updated_at"] = utc_now_iso()
        save_meta(repo, meta)

    current = meta.get("merge_prefer")
    print(f"merge_prefer = {current if current is not None else '(none)'}")


def prompt_conflict_choice(conflict_count: int) -> str:
    """Ask the user how to resolve conflicting pixels. Returns 'ours'|'theirs'|'abort'.

    Falls back to 'abort' when no interactive input is available (e.g. piped stdin).
    """
    print(f"{conflict_count} conflicting pixels (both branches changed them differently).")
    while True:
        try:
            ans = input("Resolve conflicts with [o]urs / [t]heirs / [a]bort? ").strip().lower()
        except EOFError:
            print("No interactive input available; aborting.")
            return "abort"
        if ans in ("o", "ours"):
            return "ours"
        if ans in ("t", "theirs"):
            return "theirs"
        if ans in ("a", "abort", ""):
            return "abort"
        print("Please enter 'o', 't', or 'a'.")


def cmd_merge(args):
    repo = Path(args.repo)
    meta = load_meta(repo)
    refs = load_refs(repo)
    commits, _ = parse_log(repo / LOG_NAME)

    ours = get_head_commit(refs)
    theirs, _, _ = resolve_revision(args.revision, refs, commits)

    # Persist a new default conflict preference if requested ("set it once up top").
    # Done early so it sticks even when the merge is a no-op / fast-forward.
    if args.prefer is not None:
        meta["merge_prefer"] = args.prefer
        save_meta(repo, meta)
        print(f"Saved default merge preference: {args.prefer}")

    if ours == theirs:
        print("Already up to date.")
        return

    # ancestry checks for up-to-date / fast-forward
    anc_ours = ancestor_distances(ours, commits)
    if theirs in anc_ours:
        print("Already up to date.")
        return

    ours_arr = read_central_array(repo)
    theirs_arr = materialize_from_current(ours_arr, ours, theirs, commits)

    anc_theirs = ancestor_distances(theirs, commits)
    if ours in anc_theirs:
        # fast-forward
        save_central_array(repo, theirs_arr)
        if refs["HEAD"]["type"] == "branch":
            refs["branches"][refs["HEAD"]["name"]] = theirs
        else:
            refs["HEAD"] = {"type": "detached", "commit": theirs}
        save_refs(repo, refs)
        meta["updated_at"] = utc_now_iso()
        save_meta(repo, meta)
        print(f"Fast-forward to {theirs}")
        return

    base = find_merge_base(ours, theirs, commits)
    if base is None:
        raise ValueError("No merge base found.")

    base_arr = materialize_from_current(ours_arr, ours, base, commits)

    eq_ob = np.all(ours_arr == base_arr, axis=2)
    eq_tb = np.all(theirs_arr == base_arr, axis=2)
    eq_ot = np.all(ours_arr == theirs_arr, axis=2)

    take_theirs = eq_ob & ~eq_tb
    conflicts = (~eq_ot) & (~eq_ob) & (~eq_tb)
    conflict_count = int(np.count_nonzero(conflicts))

    merged = ours_arr.copy()
    merged[take_theirs] = theirs_arr[take_theirs]

    resolution = None          # "ours" or "theirs"
    resolution_source = None   # for reporting
    if conflict_count > 0:
        # Precedence: interactive prompt > --strategy / --prefer (this run) >
        # repo default (merge_prefer) > abort.
        if args.interactive:
            if sys.stdin.isatty():
                choice = prompt_conflict_choice(conflict_count)
                if choice == "abort":
                    print("Merge aborted; no changes written.")
                    return
                resolution = choice
                resolution_source = "interactive"
            else:
                print("--interactive requires a terminal; falling back to other options.")

        if resolution is None and args.strategy is not None:
            resolution = args.strategy
            resolution_source = "--strategy"

        if resolution is None and args.prefer is not None:
            resolution = args.prefer
            resolution_source = "--prefer"

        if resolution is None:
            default_pref = meta.get("merge_prefer")
            if default_pref in ("ours", "theirs"):
                resolution = default_pref
                resolution_source = "repo default (merge_prefer)"

        if resolution is None:
            print(f"Merge has {conflict_count} conflicting pixels. Aborting.")
            print(
                "Resolve with --strategy ours|theirs, --prefer ours|theirs, "
                "--interactive, or set a default: pixelgit config <repo> "
                "--merge-prefer ours|theirs"
            )
            return

        if resolution == "theirs":
            merged[conflicts] = theirs_arr[conflicts]
        # "ours" keeps merged as-is

    head_desc = refs["HEAD"]["name"] if refs["HEAD"]["type"] == "branch" else "detached-HEAD"
    msg = args.message or f"merge {args.revision} into {head_desc}"

    cid, changed = write_new_commit(
        repo=repo,
        meta=meta,
        refs=refs,
        old_arr=ours_arr,
        new_arr=merged,
        parents=[ours, theirs],   # first parent = ours (diff parent)
        message=msg,
        allow_empty=True,         # keep topology even if no pixel delta
    )

    print(f"Merged {theirs} into {ours} -> {cid}")
    print(f"Changed pixels: {changed}")
    if conflict_count:
        print(
            f"Conflicts resolved as '{resolution}' via {resolution_source}: "
            f"{conflict_count} px"
        )


# ---------------- Lineage visualization ----------------

# Branch lane colors (GitHub-ish palette).
LANE_PALETTE = [
    (88, 166, 255),   # blue
    (63, 185, 80),    # green
    (219, 119, 40),   # orange
    (188, 140, 255),  # purple
    (247, 129, 102),  # salmon
    (219, 119, 189),  # pink
    (121, 192, 255),  # light blue
    (210, 153, 34),   # gold
]
BG_COLOR = (13, 17, 23)          # dark background
FG_COLOR = (230, 237, 243)       # primary text
MUTED_COLOR = (139, 148, 158)    # secondary text
HEAD_PILL = (56, 139, 253)       # HEAD decoration


def _load_font(size: int, bold: bool = False):
    regular = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "DejaVuSans.ttf",
        "Arial.ttf",
    ]
    boldfonts = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "DejaVuSans-Bold.ttf",
        "Arial Bold.ttf",
    ]
    for name in (boldfonts if bold else regular):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_arrow(draw, p0, p1, color, width=6, head=22):
    """Draw a line from p0 to p1 with an arrowhead at p1."""
    draw.line([p0, p1], fill=color, width=width)
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    for da in (math.radians(150), math.radians(-150)):
        hx = p1[0] + head * math.cos(ang + da)
        hy = p1[1] + head * math.sin(ang + da)
        draw.line([p1, (hx, hy)], fill=color, width=width)


def _reconstruct_image(repo, central_arr, head_commit, cid, commits) -> Image.Image:
    """Full-resolution RGB image at a commit, transparency composited on BG_COLOR."""
    arr = materialize_from_current(central_arr, head_commit, cid, commits)
    rgba = Image.fromarray(arr)  # (H, W, 4) uint8 -> RGBA
    if rgba.mode != "RGBA":
        rgba = rgba.convert("RGBA")
    bg = Image.new("RGB", rgba.size, BG_COLOR)
    bg.paste(rgba, mask=rgba.split()[3])
    return bg


def render_lineage(repo: Path, commits: dict[str, Commit], order: list[str], refs: dict, out_path: Path) -> None:
    if not order:
        raise ValueError("No commits to graph.")

    central_arr = read_central_array(repo)
    head_commit = get_head_commit(refs)
    head_is_branch = refs["HEAD"]["type"] == "branch"

    branch_heads: dict[str, list[str]] = {}
    for b, cid in refs["branches"].items():
        branch_heads.setdefault(cid, []).append(b)

    def branch_key(cid: str) -> str:
        b = commits[cid].branch
        return b if b is not None else "(detached)"

    # Assign a column (lane) per branch, in first-appearance order.
    col_of_branch: dict[str, int] = {}
    for cid in order:
        k = branch_key(cid)
        if k not in col_of_branch:
            col_of_branch[k] = len(col_of_branch)
    ncols = max(1, len(col_of_branch))

    # Newest commit at the top.
    rows = list(reversed(order))
    row_of = {cid: i for i, cid in enumerate(rows)}

    # Reconstruct every commit's full-resolution image.
    imgs = {cid: _reconstruct_image(repo, central_arr, head_commit, cid, commits) for cid in rows}
    iw, ih = imgs[rows[0]].size

    font_id = _load_font(max(22, iw // 26), bold=True)
    font_msg = _load_font(max(18, iw // 32))
    font_small = _load_font(max(16, iw // 36))
    font_title = _load_font(max(28, iw // 20), bold=True)

    measure = ImageDraw.Draw(Image.new("RGB", (4, 4)))

    def tw(txt: str, f) -> int:
        box = measure.textbbox((0, 0), txt, font=f)
        return box[2] - box[0]

    def th(txt: str, f) -> int:
        box = measure.textbbox((0, 0), txt, font=f)
        return box[3] - box[1]

    label_h = th("Ag", font_id) + 26
    border = 5
    gap_x = max(80, iw // 6)
    gap_y = max(140, ih // 6)
    left = 48
    top = th("Ag", font_title) + 48
    lane_w = iw + 2 * border + gap_x
    row_h = ih + 2 * border + label_h + gap_y

    width = int(left * 2 + (ncols - 1) * lane_w + iw + 2 * border)
    height = int(top + len(rows) * row_h)

    img = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)
    draw.text((left, 24), "PixelGit lineage", font=font_title, fill=FG_COLOR)

    def lane_color(cid: str) -> tuple[int, int, int]:
        return LANE_PALETTE[col_of_branch[branch_key(cid)] % len(LANE_PALETTE)]

    def cell_xy(cid: str) -> tuple[int, int]:
        col = col_of_branch[branch_key(cid)]
        return left + col * lane_w, top + row_of[cid] * row_h

    def image_box(cid: str) -> tuple[int, int, int, int]:
        x, y = cell_xy(cid)
        iy = y + label_h
        return x + border, iy + border, x + border + iw, iy + border + ih

    # Arrows: each commit points down to its parent(s).
    for cid in rows:
        bx0, by0, bx1, by1 = image_box(cid)
        start = ((bx0 + bx1) // 2, by1 + border)
        for p in commits[cid].parents:
            if p not in row_of:
                continue
            px0, py0, px1, py1 = image_box(p)
            end = ((px0 + px1) // 2, py0 - border)
            _draw_arrow(draw, start, end, lane_color(cid))

    # Each node: label above, then the full image with a colored branch frame.
    for cid in rows:
        c = commits[cid]
        cx, cy = cell_xy(cid)
        bx0, by0, bx1, by1 = image_box(cid)
        color = lane_color(cid)

        # colored frame (double thickness for merge commits)
        fw = border * 2 if len(c.parents) > 1 else border
        draw.rectangle([bx0 - fw, by0 - fw, bx1 + fw, by1 + fw], outline=color, width=fw)
        img.paste(imgs[cid], (bx0, by0))

        # label row above the image
        deco = list(branch_heads.get(cid, []))
        if cid == head_commit:
            deco.append("HEAD" if head_is_branch else "HEAD (detached)")
        ly = cy + 4
        draw.text((cx, ly), cid, font=font_id, fill=FG_COLOR)
        lx = cx + tw(cid, font_id) + 16
        pill_h = th("Ag", font_small) + 12
        for name in deco:
            pill = HEAD_PILL if name.startswith("HEAD") else color
            nw = tw(name, font_small)
            draw.rounded_rectangle([lx, ly, lx + nw + 20, ly + pill_h], radius=pill_h // 2, fill=pill)
            draw.text((lx + 10, ly + 5), name, font=font_small, fill=(255, 255, 255))
            lx += nw + 20 + 10
        if c.message:
            draw.text((lx + 4, ly + 3), c.message, font=font_msg, fill=MUTED_COLOR)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def cmd_graph(args):
    repo = Path(args.repo)
    refs = load_refs(repo)
    commits, order = parse_log(repo / LOG_NAME)
    out = Path(args.output) if args.output else repo / "lineage.png"
    render_lineage(repo, commits, order, refs, out)
    w, h = Image.open(out).size
    print(f"Wrote lineage graph: {out} ({len(order)} commits, {w}x{h})")


# ---------------- CLI ----------------

def build_parser():
    p = argparse.ArgumentParser(description="PixelGit: pixel-level image versioning with branching + merging")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Initialize repo with base image")
    p_init.add_argument("repo", help="Repo directory")
    p_init.add_argument("image", help="Input image path")
    p_init.add_argument(
        "--max-total-pixels",
        type=int,
        default=DEFAULT_MAX_TOTAL_PIXELS,
        help=f"Max total pixels for central image (default: {DEFAULT_MAX_TOTAL_PIXELS})",
    )
    p_init.set_defaults(func=cmd_init)

    p_commit = sub.add_parser("commit", help="Commit changes from an image")
    p_commit.add_argument("repo", help="Repo directory")
    p_commit.add_argument("image", help="Edited image path")
    p_commit.add_argument("-m", "--message", default="", help="Commit message")
    p_commit.set_defaults(func=cmd_commit)

    p_log = sub.add_parser("log", help="Show commit history")
    p_log.add_argument("repo", help="Repo directory")
    p_log.set_defaults(func=cmd_log)

    p_branch = sub.add_parser("branch", help="List/create branches")
    p_branch.add_argument("repo", help="Repo directory")
    p_branch.add_argument("name", nargs="?", help="Branch name (omit to list)")
    p_branch.add_argument("start", nargs="?", help="Start revision (branch/commit/HEAD)")
    p_branch.add_argument("-f", "--force", action="store_true", help="Move existing branch")
    p_branch.add_argument("-c", "--checkout", action="store_true", help="Checkout branch after creating/moving")
    p_branch.set_defaults(func=cmd_branch)

    p_checkout = sub.add_parser("checkout", help="Checkout branch or commit")
    p_checkout.add_argument("repo", help="Repo directory")
    p_checkout.add_argument("revision", help="Branch name, commit id, or HEAD")
    p_checkout.add_argument("--detach", action="store_true", help="Detach even if revision is a branch")
    p_checkout.set_defaults(func=cmd_checkout)

    p_merge = sub.add_parser("merge", help="Merge revision into current HEAD")
    p_merge.add_argument("repo", help="Repo directory")
    p_merge.add_argument("revision", help="Branch or commit to merge")
    p_merge.add_argument("-m", "--message", default=None, help="Merge commit message")
    p_merge.add_argument(
        "--strategy",
        choices=["ours", "theirs"],
        default=None,
        help="Conflict strategy for per-pixel conflicts (one-off, overrides repo default)",
    )
    p_merge.add_argument(
        "--prefer",
        choices=["ours", "theirs"],
        default=None,
        help="Set a persistent default conflict preference (stored in the repo) and use it for this merge",
    )
    p_merge.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Prompt for which branch to prefer when there are conflicting pixels",
    )
    p_merge.set_defaults(func=cmd_merge)

    p_graph = sub.add_parser("graph", help="Render the commit lineage to an image")
    p_graph.add_argument("repo", help="Repo directory")
    p_graph.add_argument("-o", "--output", default=None, help="Output image path (default: <repo>/lineage.png)")
    p_graph.set_defaults(func=cmd_graph)

    p_config = sub.add_parser("config", help="View or set repo configuration")
    p_config.add_argument("repo", help="Repo directory")
    p_config.add_argument(
        "--merge-prefer",
        choices=["ours", "theirs", "none"],
        default=None,
        help="Persistent default conflict preference for merges (set once 'up top')",
    )
    p_config.set_defaults(func=cmd_config)

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