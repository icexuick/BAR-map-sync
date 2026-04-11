"""
Batch fix: re-extract all maps with missing textures, commit, push, purge.
"""
import os
import json
import subprocess
import sys
import requests
import time

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
MAPS_DIR = os.path.join(REPO_ROOT, 'maps_placement')
FEATURES_DIR = os.path.join(REPO_ROOT, 'maps_features')
CDN_PURGE = 'https://purge.jsdelivr.net/gh/icexuick/BAR-map-sync@main'
GEOVENT_NAMES = {'geovent', 'geothermal', 'geovent_feature'}
BATCH_SIZE = 5  # commit+push every N maps


def find_broken_maps():
    broken = []
    for slug in sorted(os.listdir(MAPS_DIR)):
        pf = os.path.join(MAPS_DIR, slug, 'placements.json')
        if not os.path.isfile(pf):
            continue
        with open(pf) as f:
            data = json.load(f)
        names = set(p['name'] for p in data['features'] if p['name'] not in GEOVENT_NAMES)
        missing = 0
        for name in names:
            tex_dir = os.path.join(FEATURES_DIR, name)
            if not os.path.isdir(tex_dir):
                missing += 1
                continue
            files = [f for f in os.listdir(tex_dir) if slug in f]
            if not files:
                missing += 1
        if missing > 0:
            broken.append(slug)
    return broken


def extract_map(slug):
    """Run extract_map_features.py --force for this slug."""
    # Convert slug to search term: hyphens → spaces so the Webflow
    # partial-match lookup works (slugs use hyphens, map names use spaces).
    search_term = slug.replace('-', ' ')
    cmd = [sys.executable, 'extract_map_features.py', '--map', search_term, '--force']
    print(f'\n{"="*60}')
    print(f'  EXTRACTING: {slug}')
    print(f'{"="*60}')
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    return result.returncode == 0


def git_add_and_check(slug):
    """Stage features/ maps_features/ maps_placement/<slug>/ and return True if there are changes."""
    subprocess.run(['git', 'add', 'features/', 'maps_features/',
                     f'maps_placement/{slug}/placements.json'],
                    cwd=REPO_ROOT, capture_output=True)
    r = subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=REPO_ROOT)
    return r.returncode != 0  # non-zero means there ARE staged changes


def git_commit_push(slugs):
    slug_list = ', '.join(slugs)
    msg = f'Re-extract features with textures: {slug_list}\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>'
    subprocess.run(['git', 'commit', '-m', msg], cwd=REPO_ROOT)
    subprocess.run(['git', 'push'], cwd=REPO_ROOT)
    print(f'  [PUSHED] {len(slugs)} maps')


def purge_map(slug):
    """Purge placements.json and all map-specific textures from jsDelivr."""
    requests.get(f'{CDN_PURGE}/maps_placement/{slug}/placements.json')
    count = 0
    for root, dirs, files in os.walk(FEATURES_DIR):
        for f in files:
            if slug in f:
                rel = os.path.join(root, f).replace(os.sep, '/')
                # Make relative to repo root
                rel = os.path.relpath(rel, REPO_ROOT).replace(os.sep, '/')
                requests.get(f'{CDN_PURGE}/{rel}')
                count += 1
    return count


def main():
    broken = find_broken_maps()
    print(f'Found {len(broken)} maps with missing textures\n')

    batch_slugs = []
    done = 0
    failed = []

    for slug in broken:
        ok = extract_map(slug)
        if not ok:
            print(f'  [FAILED] {slug}')
            failed.append(slug)
            continue

        has_changes = git_add_and_check(slug)
        if has_changes:
            batch_slugs.append(slug)

        done += 1

        # Commit+push in batches
        if len(batch_slugs) >= BATCH_SIZE:
            git_commit_push(batch_slugs)
            for s in batch_slugs:
                n = purge_map(s)
                print(f'  [PURGED] {s}: {n} textures')
            batch_slugs = []
            time.sleep(1)

        print(f'\n  Progress: {done}/{len(broken)}')

    # Final batch
    if batch_slugs:
        git_commit_push(batch_slugs)
        for s in batch_slugs:
            n = purge_map(s)
            print(f'  [PURGED] {s}: {n} textures')

    print(f'\n{"="*60}')
    print(f'DONE: {done - len(failed)} succeeded, {len(failed)} failed')
    if failed:
        print(f'Failed maps: {", ".join(failed)}')


if __name__ == '__main__':
    main()
