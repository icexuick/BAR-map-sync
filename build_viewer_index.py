"""
Writes features/index.json — a list of every feature model that exists on
disk, so the static viewer (viewer/index.html) can render them in a grid
without needing a backend.

Run after extract_map_features.py whenever you add maps.
"""

import json
import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR = os.path.join(REPO_ROOT, "features")
INDEX_PATH = os.path.join(FEATURES_DIR, "index.json")


def main():
    if not os.path.isdir(FEATURES_DIR):
        print(f"No features/ directory at {FEATURES_DIR}")
        return

    entries = []
    for name in sorted(os.listdir(FEATURES_DIR)):
        if name.startswith('_') or name == 'index.json':
            continue
        folder = os.path.join(FEATURES_DIR, name)
        glb = os.path.join(folder, 'model.glb')
        if os.path.isdir(folder) and os.path.isfile(glb):
            entries.append({
                'name': name,
                'glb': f"{name}/model.glb",
                'size': os.path.getsize(glb),
            })

    with open(INDEX_PATH, 'w', encoding='utf-8') as f:
        json.dump({'features': entries}, f, indent=2)

    print(f"Wrote {INDEX_PATH} with {len(entries)} entries")
    total = sum(e['size'] for e in entries)
    print(f"Total GLB size: {total/1024:.0f} KB")


if __name__ == "__main__":
    main()
