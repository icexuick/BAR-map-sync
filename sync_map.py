"""
BAR Map Sync — end-to-end sync for one map

Pipeline:
  1. Run extract_map_features.py for the requested map (downloads sd7,
     parses set.lua, builds GLBs that don't exist yet, writes placements.json
     + heightmap.png).
  2. Show unresolved features (if any) and ask the user to confirm.
  3. Inspect git to see whether the extract produced new feature GLBs/textures
     under features/, or just refreshed maps_features/<slug>/placements.json.
  4. Two-phase commit/push when there are new features:
        Phase 1: commit + push everything under features/  ->  jsDelivr purge
                 of one new GLB to force a refresh.
        Phase 2: commit + push maps_features/<slug>/placements.json  ->
                 jsDelivr purge of placements.json so the live site picks it
                 up immediately.
     This ordering matters: jsDelivr caches 404s for up to 24h, so the
     placements.json (which references the new GLB URLs) MUST go live AFTER
     the GLBs themselves are reachable.
  5. Fast path when there are no new features: a single commit + push of
     placements.json + a purge.

Usage:
    python sync_map.py "altored divide"
    python sync_map.py "altair crossing" --yes      # skip prompts

Notes:
  - Heightmaps are gitignored (maps_features/**/*.png) so only placements.json
    ever goes to GitHub. The PNGs stay local for viewer/map.html.
  - This script only stages paths it produced (features/, maps_features/...).
    Other dirty files in the working tree are left alone.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from typing import List, Optional, Tuple

import requests


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
GH_OWNER = "icexuick"
GH_REPO = "BAR-map-sync"
GH_BRANCH = "main"
JSDELIVR_PURGE_BASE = f"https://purge.jsdelivr.net/gh/{GH_OWNER}/{GH_REPO}@{GH_BRANCH}"
JSDELIVR_CDN_BASE = f"https://cdn.jsdelivr.net/gh/{GH_OWNER}/{GH_REPO}@{GH_BRANCH}"


# -- shell helpers ----------------------------------------------------------

def run(cmd: List[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """Thin wrapper around subprocess.run with sensible defaults. Streams the
    child's stdout/stderr live unless capture=True (in which case the caller
    wants to read it back)."""
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=check,
        text=True,
        capture_output=capture,
    )


def git_porcelain(paths: List[str]) -> List[Tuple[str, str]]:
    """Return [(status_code, path), ...] for `git status --porcelain` over the
    given paths. Status code is the 2-char prefix (e.g. ' M', '??', 'A ').
    Empty list means clean."""
    cp = run(["git", "status", "--porcelain", "--"] + paths, capture=True)
    out: List[Tuple[str, str]] = []
    for line in cp.stdout.splitlines():
        if not line:
            continue
        # Porcelain v1 format: XY<space>path
        code = line[:2]
        path = line[3:]
        out.append((code, path))
    return out


# -- jsDelivr ---------------------------------------------------------------

def jsdelivr_purge(path: str) -> bool:
    """Force jsDelivr to drop its cached copy of one file. Returns True on
    success. Used after each push so the live site doesn't have to wait the
    default ~12h cache TTL.

    The purge endpoint is GET-based: a 200 with `{"success": true}` means it
    worked. We don't fail the whole sync on a purge error — the file is still
    in git, jsDelivr will catch up on its own eventually."""
    url = f"{JSDELIVR_PURGE_BASE}/{path.lstrip('/')}"
    print(f"  purge: {url}")
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            try:
                body = r.json()
                if body.get("status") == "finished" or body.get("success"):
                    print(f"    [OK] purged")
                    return True
                print(f"    [!] purge response: {body}")
                return False
            except ValueError:
                print(f"    [!] purge response not JSON ({len(r.text)} bytes)")
                return False
        print(f"    [!] purge HTTP {r.status_code}")
        return False
    except requests.RequestException as e:
        print(f"    [!] purge failed: {e}")
        return False


# -- extractor --------------------------------------------------------------

def run_extractor(map_filter: str) -> None:
    """Invoke extract_map_features.py as a subprocess. Streams output live so
    the user sees the same progress they'd see running it directly."""
    print(f"\n=== [1/3] Extracting features for \"{map_filter}\" ===")
    run([sys.executable, "extract_map_features.py", "--map", map_filter])


def find_extracted_slug(map_filter: str) -> Optional[str]:
    """Find which maps_placement/<slug>/placements.json was just produced.
    Strategy: look for the most recently modified placements.json under
    maps_placement/. The extractor uses fuzzy matching on Webflow names so we
    can't reliably re-derive the slug from the user's input."""
    maps_root = os.path.join(REPO_ROOT, "maps_placement")
    if not os.path.isdir(maps_root):
        return None
    candidates: List[Tuple[float, str]] = []
    for entry in os.listdir(maps_root):
        p = os.path.join(maps_root, entry, "placements.json")
        if os.path.isfile(p):
            candidates.append((os.path.getmtime(p), entry))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def load_placements(slug: str) -> dict:
    p = os.path.join(REPO_ROOT, "maps_placement", slug, "placements.json")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# -- summary helpers --------------------------------------------------------

def summarize_extracted(slug: str) -> dict:
    """Read placements.json and return a small summary dict for logging."""
    data = load_placements(slug)
    feats = data.get("features", [])
    geovents = [f for f in feats if f.get("name") == "geovent"]
    return {
        "name": data.get("name"),
        "slug": slug,
        "total": len(feats),
        "geovents": len(geovents),
        "world": (data.get("worldWidth"), data.get("worldHeight")),
        "height_range": (data.get("minHeight"), data.get("maxHeight")),
    }


def confirm(prompt: str, assume_yes: bool) -> bool:
    if assume_yes:
        print(f"{prompt} [auto-yes]")
        return True
    try:
        ans = input(f"{prompt} [y/N] ").strip().lower()
    except EOFError:
        return False
    return ans in ("y", "yes")


# -- git phases -------------------------------------------------------------

def stage_features(features_changes: List[Tuple[str, str]]) -> List[str]:
    """Stage every changed/new path under features/ as reported by porcelain.
    Returns the list of paths actually added."""
    paths = [p for _, p in features_changes]
    if not paths:
        return []
    run(["git", "add", "--"] + paths)
    return paths


def commit(message: str) -> bool:
    """Create a commit. Returns False if there was nothing staged."""
    cp = run(["git", "commit", "-m", message], check=False, capture=True)
    if cp.returncode == 0:
        print(cp.stdout.strip())
        return True
    out = (cp.stdout or "") + (cp.stderr or "")
    if "nothing to commit" in out or "no changes added" in out:
        print("  (nothing to commit)")
        return False
    # Real error — surface it.
    sys.stdout.write(out)
    raise SystemExit(f"git commit failed (exit {cp.returncode})")


def push() -> None:
    run(["git", "push"])


def pick_purge_target(features_changes: List[Tuple[str, str]]) -> Optional[str]:
    """Pick one newly-added GLB to purge as a representative. We don't purge
    every individual file — purging one is enough to nudge jsDelivr into
    refreshing its directory listing for the affected paths."""
    for _code, path in features_changes:
        if path.endswith("model.glb"):
            return path
    # Fall back to anything under features/
    for _code, path in features_changes:
        if path.startswith("features/"):
            return path
    return None


# -- main flow --------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Extract + commit + push one BAR map")
    ap.add_argument("map", help='Map name (fuzzy match against Webflow CMS), e.g. "altored divide"')
    ap.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompts")
    ap.add_argument("--skip-extract", action="store_true",
                    help="Skip running extract_map_features.py (use already-generated files)")
    args = ap.parse_args()

    # 1. Extract
    if not args.skip_extract:
        run_extractor(args.map)
    else:
        print("=== [1/3] Skipping extractor (--skip-extract) ===")

    # 2. Find what was just produced
    slug = find_extracted_slug(args.map)
    if not slug:
        raise SystemExit("Could not find any maps_placement/<slug>/placements.json — extract failed?")
    summary = summarize_extracted(slug)
    print(f"\nExtracted: {summary['name']!r}")
    print(f"  slug:         {summary['slug']}")
    print(f"  world:        {summary['world'][0]}x{summary['world'][1]} elmos")
    print(f"  height range: {summary['height_range'][0]}..{summary['height_range'][1]}")
    print(f"  placements:   {summary['total']}")
    print(f"  geovents:     {summary['geovents']}")

    # 3. Inspect git for what changed
    print(f"\n=== [2/3] Checking git for new content ===")
    feature_changes = git_porcelain(["features/"])
    placement_path = f"maps_placement/{slug}/placements.json"
    placement_changes = git_porcelain([placement_path])

    print(f"  features/ changes:    {len(feature_changes)}")
    print(f"  placements.json:      {'modified' if placement_changes else 'unchanged'}")

    if not feature_changes and not placement_changes:
        print("\nNothing changed — nothing to push. Done.")
        return

    # 4. Confirm
    if feature_changes:
        # Show a quick preview of which feature dirs are new
        new_dirs = set()
        for _code, path in feature_changes:
            parts = path.split("/")
            if len(parts) >= 2 and parts[0] == "features":
                new_dirs.add(parts[1])
        print(f"\n  new/changed feature dirs ({len(new_dirs)}):")
        for d in sorted(new_dirs)[:20]:
            print(f"    - {d}")
        if len(new_dirs) > 20:
            print(f"    ... +{len(new_dirs) - 20} more")

    if not confirm("\nProceed with commit + push?", args.yes):
        print("Aborted.")
        return

    print(f"\n=== [3/3] Committing & pushing ===")

    # Phase 1: features (only if there are any)
    if feature_changes:
        print("\n-- Phase 1: features --")
        stage_features(feature_changes)
        commit_msg = f'Add features for "{summary["name"]}"'
        if commit(commit_msg):
            push()
            target = pick_purge_target(feature_changes)
            if target:
                jsdelivr_purge(target)

    # Phase 2: placements (always — even if it was unchanged we want to ensure
    # it lines up with the GLBs we just pushed; but git will skip the commit
    # if it's already in sync).
    print("\n-- Phase 2: placements --")
    run(["git", "add", "--", placement_path])
    commit_msg = (
        f'Sync placements for "{summary["name"]}" '
        f'({summary["total"]} placements, {summary["geovents"]} geovents)'
        if feature_changes else
        f'Re-sync placements for "{summary["name"]}" '
        f'({summary["total"]} placements)'
    )
    if commit(commit_msg):
        push()
        jsdelivr_purge(placement_path)

    # 5. Final summary
    live_url = f"{JSDELIVR_CDN_BASE}/{placement_path}"
    print(f"\nDone. Live placements:\n  {live_url}")


if __name__ == "__main__":
    main()
