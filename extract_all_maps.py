"""
Batch-extract features for ALL maps from Webflow.
Skips maps that already have a placements.json in maps_placement/.
Logs progress and errors to extract_all_maps.log.
"""
import os, sys, subprocess, requests, time
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PLACEMENTS_DIR = os.path.join(REPO_ROOT, "maps_placement")
WEBFLOW_API_TOKEN = os.environ.get("WEBFLOW_API_TOKEN")
COLLECTION_ID = "6564c6553676389f8ba45aaf"
HEADERS = {"Authorization": f"Bearer {WEBFLOW_API_TOKEN}", "accept": "application/json"}
LOG_FILE = os.path.join(REPO_ROOT, "extract_all_maps.log")


def fetch_all_map_names():
    url = f"https://api.webflow.com/v2/collections/{COLLECTION_ID}/items"
    offset = 0
    names = []
    while True:
        r = requests.get(url, headers=HEADERS, params={"limit": 100, "offset": offset})
        r.raise_for_status()
        data = r.json()
        batch = data.get("items", [])
        if not batch:
            break
        for item in batch:
            name = item.get("fieldData", {}).get("name", "")
            if name:
                names.append(name)
        if len(batch) < 100:
            break
        offset += 100
    return sorted(names, key=str.lower)


def slugify(name):
    return name.lower().replace(" ", "-").replace("'", "").replace(",", "")


def already_done(name):
    slug = slugify(name)
    return os.path.isfile(os.path.join(PLACEMENTS_DIR, slug, "placements.json"))


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    all_maps = fetch_all_map_names()
    todo = [m for m in all_maps if not already_done(m)]
    log(f"Total maps: {len(all_maps)}, already done: {len(all_maps) - len(todo)}, todo: {len(todo)}")

    failed = []
    for i, name in enumerate(todo, 1):
        log(f"\n{'='*60}")
        log(f"[{i}/{len(todo)}] Extracting: {name}")
        log(f"{'='*60}")
        try:
            result = subprocess.run(
                [sys.executable, "extract_map_features.py", "--map", name],
                cwd=REPO_ROOT,
                timeout=600,  # 10 min per map max
            )
            if result.returncode != 0:
                log(f"FAILED (exit {result.returncode}): {name}")
                failed.append((name, f"exit {result.returncode}"))
            else:
                log(f"OK: {name}")
        except subprocess.TimeoutExpired:
            log(f"TIMEOUT: {name}")
            failed.append((name, "timeout"))
        except Exception as e:
            log(f"ERROR: {name} - {e}")
            failed.append((name, str(e)))

    log(f"\n{'='*60}")
    log(f"DONE. Processed {len(todo)} maps, {len(failed)} failed.")
    if failed:
        log("Failed maps:")
        for name, reason in failed:
            log(f"  - {name}: {reason}")


if __name__ == "__main__":
    main()
