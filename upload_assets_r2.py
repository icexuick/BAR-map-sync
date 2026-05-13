"""
Bulk-upload BAR-Map-Sync assets to Cloudflare R2.

Uploads three local directories (mirrored as-is in the bucket):
    maps_features/
    maps_placement/
    features/

Plus two one-off external assets we want to host ourselves:
    assets/hdr/clarens_midday_1k.hdr      (downloaded from BAR-modelviewer repo)
    assets/waterbump_4tiles.png           (downloaded from map_blueprint repo)

Skip-if-unchanged: each file's local MD5 is compared against the remote ETag.
Files that already match are skipped, so re-running this script after small
changes is fast.

Usage:
    python upload_assets_r2.py                    # sync everything
    python upload_assets_r2.py --only features    # sync just one dir
    python upload_assets_r2.py --dry-run          # show what would happen
    python upload_assets_r2.py --workers 16       # parallelism (default 8)
"""

import argparse
import io
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, List, Optional, Tuple

import requests

from r2_client import (
    R2_BUCKET,
    content_type_for,
    head_etag,
    make_client,
    md5_of_file,
    public_url,
    upload_file,
)


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Directories to mirror 1:1 from disk to bucket.
SYNC_DIRS = ["maps_features", "maps_placement", "features"]

# One-off external assets: (remote_url, target_key)
EXTERNAL_ASSETS = [
    (
        "https://raw.githubusercontent.com/icexuick/BAR-modelviewer/main/hdr/clarens_midday_1k.hdr",
        "assets/hdr/clarens_midday_1k.hdr",
    ),
    (
        "https://raw.githubusercontent.com/beyond-all-reason/map_blueprint/a7db394716730e102f84442ed35fe3f7644a01fc/maps/waterbump_4tiles.png",
        "assets/waterbump_4tiles.png",
    ),
]


def iter_local_files(root_dir: str) -> Iterable[Tuple[str, str]]:
    """Yield (local_path, key) for every file under root_dir.
    Key is the path relative to REPO_ROOT with forward slashes."""
    abs_root = os.path.join(REPO_ROOT, root_dir)
    if not os.path.isdir(abs_root):
        return
    for dirpath, _dirnames, filenames in os.walk(abs_root):
        for fn in filenames:
            local = os.path.join(dirpath, fn)
            rel = os.path.relpath(local, REPO_ROOT).replace(os.sep, "/")
            yield local, rel


def upload_one(client, local_path: str, key: str, dry_run: bool) -> Tuple[str, str]:
    """Upload a single file. Returns (status, key) where status is one of
    'uploaded' | 'skipped' | 'would-upload' | 'would-skip'."""
    if dry_run:
        local_md5 = md5_of_file(local_path)
        remote_etag = head_etag(client, key)
        if remote_etag and remote_etag == local_md5:
            return ("would-skip", key)
        return ("would-upload", key)
    status = upload_file(client, local_path, key, skip_if_unchanged=True)
    return (status, key)


def sync_directory(client, root_dir: str, workers: int, dry_run: bool) -> dict:
    files = list(iter_local_files(root_dir))
    if not files:
        print(f"  {root_dir}: nothing to upload (directory missing or empty)")
        return {"total": 0, "uploaded": 0, "skipped": 0, "errors": 0}

    total = len(files)
    total_bytes = sum(os.path.getsize(p) for p, _ in files)
    print(f"  {root_dir}: {total} files, {total_bytes / 1024 / 1024:.1f} MB")

    uploaded = skipped = errors = 0
    completed = 0
    last_progress_pct = -1

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(upload_one, client, p, k, dry_run): (p, k) for p, k in files}
        for fut in as_completed(futures):
            p, k = futures[fut]
            try:
                status, _ = fut.result()
                if status in ("uploaded", "would-upload"):
                    uploaded += 1
                elif status in ("skipped", "would-skip"):
                    skipped += 1
            except Exception as e:
                errors += 1
                print(f"    [!] {k}: {e}")
            completed += 1
            pct = (completed * 100) // total
            if pct != last_progress_pct and pct % 5 == 0:
                print(f"    progress: {completed}/{total} ({pct}%) — "
                      f"up={uploaded} skip={skipped} err={errors}")
                last_progress_pct = pct

    print(f"  {root_dir}: done — uploaded={uploaded}, skipped={skipped}, errors={errors}")
    return {"total": total, "uploaded": uploaded, "skipped": skipped, "errors": errors}


def upload_external(client, dry_run: bool) -> dict:
    """Download external assets and push them to R2 under assets/."""
    uploaded = skipped = errors = 0
    for url, key in EXTERNAL_ASSETS:
        try:
            print(f"  external: {key} <- {url}")
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            body = r.content
            # Compare against remote ETag using MD5 of bytes
            import hashlib
            local_md5 = hashlib.md5(body).hexdigest()
            remote_etag = head_etag(client, key)
            if remote_etag and remote_etag == local_md5:
                print(f"    skipped (unchanged, {len(body)} bytes)")
                skipped += 1
                continue
            if dry_run:
                print(f"    would-upload ({len(body)} bytes)")
                uploaded += 1
                continue
            client.put_object(
                Bucket=R2_BUCKET,
                Key=key,
                Body=io.BytesIO(body),
                ContentType=content_type_for(key),
                CacheControl="public, max-age=604800",
            )
            print(f"    uploaded ({len(body)} bytes) -> {public_url(key)}")
            uploaded += 1
        except Exception as e:
            print(f"    [!] failed: {e}")
            errors += 1
    return {"uploaded": uploaded, "skipped": skipped, "errors": errors}


def main():
    ap = argparse.ArgumentParser(description="Sync local assets to Cloudflare R2")
    ap.add_argument("--only", choices=SYNC_DIRS + ["external"], action="append",
                    help="Only sync this directory (or 'external'). Can repeat.")
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel upload workers (default 8)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be uploaded without writing")
    args = ap.parse_args()

    targets = args.only or (SYNC_DIRS + ["external"])
    print(f"R2 bucket: {R2_BUCKET}")
    print(f"workers:   {args.workers}")
    print(f"dry-run:   {args.dry_run}")
    print(f"targets:   {targets}")
    print()

    client = make_client()

    totals = {"uploaded": 0, "skipped": 0, "errors": 0}
    for target in targets:
        print(f"=== {target} ===")
        if target == "external":
            r = upload_external(client, args.dry_run)
        else:
            r = sync_directory(client, target, args.workers, args.dry_run)
        for k in ("uploaded", "skipped", "errors"):
            totals[k] += r.get(k, 0)
        print()

    print(f"All done. uploaded={totals['uploaded']}, skipped={totals['skipped']}, errors={totals['errors']}")
    if totals["errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
