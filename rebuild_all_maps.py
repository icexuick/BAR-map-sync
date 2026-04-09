"""One-shot helper: re-extracts every Webflow map that has a downloadurl,
with --force, so all features are regenerated under the current pipeline.

Stops early once no stale features remain (stale = GLBs older than the
glb_feature_builder.py file). Skips maps that fail.

Intended for one-time cleanup after a pipeline change — not something you
should run regularly.
"""

import os
import subprocess
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_map_features import HEADERS_WEBFLOW, COLLECTION_ID, FIELD_DOWNLOAD_URL


def list_maps():
    url = f"https://api.webflow.com/v2/collections/{COLLECTION_ID}/items"
    offset = 0
    limit = 100
    out = []
    while True:
        r = requests.get(url, headers=HEADERS_WEBFLOW,
                         params={'limit': limit, 'offset': offset})
        r.raise_for_status()
        data = r.json()
        batch = data.get('items', [])
        if not batch:
            break
        for item in batch:
            name = item.get('fieldData', {}).get('name', '')
            dl = item.get('fieldData', {}).get(FIELD_DOWNLOAD_URL)
            if name and dl:
                out.append(name)
        if len(batch) < limit:
            break
        offset += limit
    return out


def count_stale(cutoff_mtime: float) -> int:
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'features')
    stale = 0
    for name in os.listdir(root):
        if name.startswith('_') or name == 'index.json':
            continue
        glb = os.path.join(root, name, 'model.glb')
        if os.path.isfile(glb) and os.path.getmtime(glb) < cutoff_mtime:
            stale += 1
    return stale


def main():
    cutoff = os.path.getmtime(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'glb_feature_builder.py')) - 60
    stale_before = count_stale(cutoff)
    print(f"Stale features at start: {stale_before}")
    if stale_before == 0:
        print("Nothing to do.")
        return

    maps = list_maps()
    print(f"Webflow maps to try: {len(maps)}")

    for i, name in enumerate(maps, 1):
        stale = count_stale(cutoff)
        if stale == 0:
            print(f"[{i}/{len(maps)}] All features are fresh — stopping early.")
            break
        print(f"\n[{i}/{len(maps)}] {name}  (stale={stale})")
        try:
            result = subprocess.run(
                [sys.executable, 'extract_map_features.py', '--map', name, '--force'],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                capture_output=True, text=True, timeout=600,
            )
            # Print a short summary
            out = result.stdout.splitlines()
            for line in out[-6:]:
                print(f"    {line}")
            if result.returncode != 0:
                print(f"    [!] exit {result.returncode}")
        except subprocess.TimeoutExpired:
            print("    [!] timed out")
        except Exception as e:
            print(f"    [!] error: {e}")

    stale_after = count_stale(cutoff)
    print(f"\nStale at end: {stale_after}  (was {stale_before})")


if __name__ == "__main__":
    main()
