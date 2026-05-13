"""
Extract terrain textures (diffuse / height / metal / normal / skybox) from a
BAR map's SD7 archive and save them locally as WebP files.

This is the same texture-set that update_maps_local.py currently uploads via
FTP, but written to disk instead so you can inspect what gets generated. No
FTP, no Webflow API calls.

Usage:
    python extract_map_textures.py "ascendancy"
    python extract_map_textures.py "altored divide" --out maps_textures
"""

import argparse
import io
import json
import os
import re
import shutil
import sys
import tempfile

import py7zr
import requests
from PIL import Image

from update_maps_local import (
    DIFFUSE_MAX_WIDTH,
    MAX_FILE_SIZE_MB,
    MAX_PIXEL_DIMENSION,
    SKYBOX_MAX_WIDTH,
    SkyboxProcessor,
    SpringMapParser,
)


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(REPO_ROOT, "maps_cache.json")


def load_cache():
    if not os.path.isfile(CACHE_FILE):
        return {}
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def find_cache_entry(cache: dict, query: str):
    """Match against the cache's map names (case-insensitive substring)."""
    q = query.lower()
    matches = []
    for name, entry in cache.items():
        if q in name.lower():
            matches.append((name, entry))
    if not matches:
        return None
    if len(matches) > 1:
        # Prefer exact (case-insensitive) match if present
        exact = [m for m in matches if m[0].lower() == q]
        if exact:
            return exact[0]
        print(f"  [!] Multiple matches for {query!r}:")
        for n, _ in matches[:10]:
            print(f"        - {n}")
        print(f"      Using first: {matches[0][0]}")
    return matches[0]


def download_sd7(url: str, dest: str) -> int:
    print(f"  -> Downloading: {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                total += len(chunk)
    print(f"     ({total / 1024 / 1024:.1f} MB)")
    return total


def save_webp(img: Image.Image, out_path: str, suffix: str):
    """Same size-budget logic as update_maps_local.save_and_upload_pil_image,
    except we save to disk instead of uploading to R2.

    See that function's docstring for the full rationale; short version:
    height and metal go lossless (data textures sampled by the viewer),
    everything else goes lossy at q=85 (display textures)."""
    if img.width > MAX_PIXEL_DIMENSION or img.height > MAX_PIXEL_DIMENSION:
        print(f"     [resize] {suffix}: {img.width}x{img.height} -> max {MAX_PIXEL_DIMENSION}")
        img.thumbnail((MAX_PIXEL_DIMENSION, MAX_PIXEL_DIMENSION), Image.Resampling.LANCZOS)

    # Lossless for data textures (height, metal); lossy q=85 for the rest.
    use_lossless = suffix in ("height", "metal")
    quality = 85           # ignored by Pillow when lossless=True
    resize_factor = 1.0
    current_img = img
    while True:
        buf = io.BytesIO()
        if resize_factor < 1.0:
            # Re-derive from the original img each iteration so we don't
            # compound blur across successive shrinks.
            w = int(img.width * resize_factor)
            h = int(img.height * resize_factor)
            current_img = img.resize((w, h), Image.Resampling.BILINEAR)
        # Pillow routes to libwebp's lossless encoder when lossless=True;
        # `quality` becomes a no-op in that mode.
        current_img.save(buf, format="WEBP", quality=quality, lossless=use_lossless)
        size_mb = buf.tell() / (1024 * 1024)
        print(f"     [{suffix}] {current_img.width}x{current_img.height} q={quality} -> {size_mb:.2f} MB")
        if size_mb <= MAX_FILE_SIZE_MB:
            with open(out_path, "wb") as f:
                f.write(buf.getbuffer())
            return out_path
        # Too big — shrink. Lossless can only resort to downscaling
        # (quality has no effect); lossy drops quality first, then size.
        if use_lossless:
            resize_factor *= 0.75
        else:
            if quality > 50:
                quality -= 10
            else:
                resize_factor *= 0.75
        if resize_factor < 0.1:
            print(f"     [!] gave up on {suffix} (cannot fit budget)")
            return None


def extract_textures(map_name: str, sd7_url: str, out_dir: str, slug: str):
    os.makedirs(out_dir, exist_ok=True)
    work = tempfile.mkdtemp(prefix="bar_textures_")
    sd7 = os.path.join(work, "map.sd7")
    try:
        download_sd7(sd7_url, sd7)
        with py7zr.SevenZipFile(sd7, mode="r") as z:
            all_files = z.getnames()

            # Find mapinfo.lua for normal/skybox refs. Prefer the root-level
            # one — many maps ship a stub maphelper/mapinfo.lua that lacks
            # the real detailNormalTex / skyBox references.
            mapinfo_candidates = [f for f in all_files if f.lower().endswith("mapinfo.lua")]
            root_mapinfos = [f for f in mapinfo_candidates if f.lower() == "mapinfo.lua"]
            mapinfo_file = (root_mapinfos[0] if root_mapinfos else
                            (mapinfo_candidates[0] if mapinfo_candidates else None))
            mapinfo_content = ""
            if mapinfo_file:
                z.extract(targets=[mapinfo_file], path=work)
                with open(os.path.join(work, mapinfo_file), "r", encoding="utf-8", errors="ignore") as f:
                    mapinfo_content = f.read()

            # --- Normal map ---
            nm_match = re.search(r'detailNormalTex\s*=\s*["\']([^"\']+)["\']', mapinfo_content, re.IGNORECASE)
            if nm_match:
                nm_filename = nm_match.group(1)
                target = next((f for f in all_files if f.lower().endswith(nm_filename.lower())), None)
                if target:
                    z.extract(targets=[target], path=work)
                    try:
                        img = Image.open(os.path.join(work, target)); img.load()
                        if img.convert("L").getextrema()[1] == 0:
                            print(f"     [normal] skipped (image is fully black)")
                        else:
                            save_webp(img, os.path.join(out_dir, f"{slug}_normal.webp"), "normal")
                    except Exception as e:
                        print(f"     [normal] error: {e}")

            # --- Skybox ---
            sb_match = re.search(r'skyBox\s*=\s*["\']([^"\']+)["\']', mapinfo_content, re.IGNORECASE)
            if sb_match:
                sb_filename = sb_match.group(1)
                target = next((f for f in all_files if f.lower().endswith(sb_filename.lower())), None)
                if target:
                    z.extract(targets=[target], path=work)
                    raw_path = os.path.join(work, target)
                    img = SkyboxProcessor.process_dds_to_equi(raw_path, target_width=SKYBOX_MAX_WIDTH)
                    if img:
                        save_webp(img, os.path.join(out_dir, f"{slug}_sky.webp"), "sky")

            # --- SMF / SMT: diffuse + heightmap + metalmap ---
            smf_files = [f for f in all_files if f.lower().endswith(".smf")]
            smt_files = [f for f in all_files if f.lower().endswith(".smt")]
            targets = smf_files + smt_files
            if targets:
                print(f"     [smf/smt] extracting {len(targets)} files...")
                z.extract(targets=targets, path=work)
                smt_hint = SpringMapParser.get_filenames_from_lua(mapinfo_content) if mapinfo_content else None
                tasks = {"diffuse": True, "height": True, "metal": True}
                map_images = SpringMapParser.process_map_files(work, mapinfo_file, tasks, smt_hint)

                if "diffuse" in map_images:
                    save_webp(map_images["diffuse"], os.path.join(out_dir, f"{slug}_texture.webp"), "texture")
                if "heightmap" in map_images:
                    save_webp(map_images["heightmap"], os.path.join(out_dir, f"{slug}_height.webp"), "height")
                if "metalmap" in map_images:
                    save_webp(map_images["metalmap"], os.path.join(out_dir, f"{slug}_metal.webp"), "metal")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def main():
    ap = argparse.ArgumentParser(description="Extract terrain textures from a BAR map's SD7 archive to local WebP files")
    ap.add_argument("map", help='Map name (substring match against maps_cache.json), e.g. "ascendancy"')
    ap.add_argument("--out", default="maps_textures",
                    help="Output directory (default: maps_textures/<slug>/)")
    args = ap.parse_args()

    cache = load_cache()
    if not cache:
        print("No maps_cache.json found. Run update_maps_local.py at least once to populate the cache.")
        sys.exit(1)

    match = find_cache_entry(cache, args.map)
    if not match:
        print(f"No map matching {args.map!r} in cache.")
        sys.exit(1)

    name, entry = match
    sd7_url = entry.get("sd7_url")
    if not sd7_url:
        print(f"Cache entry for {name!r} has no sd7_url.")
        sys.exit(1)

    slug = slugify(name)
    out_dir = os.path.join(REPO_ROOT, args.out, slug)
    print(f"\n=== Extracting textures for {name!r} ===")
    print(f"  slug:    {slug}")
    print(f"  sd7:     {sd7_url}")
    print(f"  output:  {out_dir}\n")

    extract_textures(name, sd7_url, out_dir, slug)

    print(f"\nDone. Files in {out_dir}:")
    if os.path.isdir(out_dir):
        for f in sorted(os.listdir(out_dir)):
            size = os.path.getsize(os.path.join(out_dir, f)) / 1024
            print(f"  {f}  ({size:.1f} KB)")


if __name__ == "__main__":
    main()
