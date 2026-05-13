"""
BAR Map Sync — end-to-end sync for one map

Pipeline:
  1. Run extract_map_features.py for the requested map (downloads sd7,
     parses set.lua, builds GLBs that don't exist yet, writes placements.json
     + heightmap.png).
  2. Show unresolved features (if any) and ask the user to confirm.
  3. Inspect git to see whether the extract produced new feature GLBs/textures
     under features/, or just refreshed maps_features/<slug>/placements.json.
  4. Upload changed assets to Cloudflare R2 (serving CDN) and commit/push
     them to git (history/backup). Two-phase ordering when there are new
     features:
        Phase 1: upload + push everything under features/ and maps_features/
                 so the GLB/texture URLs are live before placements.json
                 references them.
        Phase 2: upload + push maps_placement/<slug>/placements.json.
  5. Fast path when there are no new features: a single upload + commit + push
     of placements.json.

Usage:
    python sync_map.py "altored divide"
    python sync_map.py "altair crossing" --yes      # skip prompts

Notes:
  - Heightmaps are gitignored (maps_features/**/*.png) so only placements.json
    ever goes to GitHub. The PNGs stay local for viewer/map.html.
  - This script only stages paths it produced (features/, maps_features/...).
    Other dirty files in the working tree are left alone.
  - R2 is the serving CDN (viewer fetches from there). Git remains the
    history/backup. If R2 upload fails, the sync stops before pushing to git
    so the two stay in sync.
"""

import argparse
import json
import os
import subprocess
import sys
from typing import List, Optional, Tuple

from r2_client import make_client, public_url, upload_file


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


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


# -- R2 upload --------------------------------------------------------------

def _r2_client_cached():
    """Lazily build & cache one R2 client per sync run."""
    if not hasattr(_r2_client_cached, "_client"):
        _r2_client_cached._client = make_client()
    return _r2_client_cached._client


def upload_paths_to_r2(paths: List[str]) -> int:
    """Upload a list of repo-relative paths to R2. Returns the count of files
    actually uploaded (skip-if-unchanged means re-runs are cheap).

    Skips paths that don't exist on disk (e.g. deleted files in porcelain
    output) and prints a one-line summary per file."""
    if not paths:
        return 0
    client = _r2_client_cached()
    uploaded = 0
    skipped = 0
    for rel in paths:
        local = os.path.join(REPO_ROOT, rel)
        if not os.path.isfile(local):
            continue
        key = rel.replace(os.sep, "/")
        try:
            status = upload_file(client, local, key, skip_if_unchanged=True)
        except Exception as e:
            print(f"    [!] R2 upload failed for {key}: {e}")
            raise
        if status == "uploaded":
            uploaded += 1
        else:
            skipped += 1
    print(f"  R2: uploaded={uploaded}, skipped={skipped} (of {len(paths)} paths)")
    return uploaded


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


# -- main flow --------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Extract + upload + commit/push one BAR map")
    ap.add_argument("map", help='Map name (fuzzy match against Webflow CMS), e.g. "altored divide"')
    ap.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompts")
    ap.add_argument("--skip-extract", action="store_true",
                    help="Skip running extract_map_features.py (use already-generated files)")
    ap.add_argument("--skip-git", action="store_true",
                    help="Upload to R2 only; skip git commit/push (R2 is the serving CDN)")
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
    map_feature_changes = git_porcelain(["maps_features/"])
    placement_path = f"maps_placement/{slug}/placements.json"
    placement_changes = git_porcelain([placement_path])

    print(f"  features/ changes:       {len(feature_changes)}")
    print(f"  maps_features/ changes:  {len(map_feature_changes)}")
    print(f"  placements.json:         {'modified' if placement_changes else 'unchanged'}")

    if not feature_changes and not map_feature_changes and not placement_changes:
        print("\nNothing changed — nothing to upload or push. Done.")
        return

    # 4. Confirm
    if feature_changes:
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

    action = "upload to R2" + ("" if args.skip_git else " + commit & push to git")
    if not confirm(f"\nProceed with {action}?", args.yes):
        print("Aborted.")
        return

    print(f"\n=== [3/3] Uploading to R2 & committing ===")

    # Phase 1: features + maps_features
    # Upload to R2 BEFORE committing so the GLBs/textures are live before
    # placements.json references them.
    asset_changes = feature_changes + map_feature_changes
    if asset_changes:
        print("\n-- Phase 1: features + maps_features --")
        asset_paths = [p for _, p in asset_changes]
        upload_paths_to_r2(asset_paths)
        if not args.skip_git:
            run(["git", "add", "--"] + asset_paths)
            commit_msg = f'Add features for "{summary["name"]}"'
            if commit(commit_msg):
                push()

    # Phase 2: placements
    print("\n-- Phase 2: placements --")
    upload_paths_to_r2([placement_path])
    if not args.skip_git:
        run(["git", "add", "--", placement_path])
        commit_msg = (
            f'Sync placements for "{summary["name"]}" '
            f'({summary["total"]} placements, {summary["geovents"]} geovents)'
            if asset_changes else
            f'Re-sync placements for "{summary["name"]}" '
            f'({summary["total"]} placements)'
        )
        if commit(commit_msg):
            push()

    # 5. Final summary
    print(f"\nDone. Live placements:\n  {public_url(placement_path)}")


if __name__ == "__main__":
    main()
