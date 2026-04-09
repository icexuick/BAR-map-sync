"""
BAR Map Features Extractor

Given a BAR map (from Webflow CMS), this script:
  1. Downloads the map sd7 archive
  2. Parses mapconfig/featureplacer/set.lua to get {name, x, z, rot} placements
  3. Resolves each unique feature name to an .s3o model + textures, looking
     first inside the sd7's own features/ + objects3d/ + unittextures/, then
     falling back to the local BAR.sdd installation
  4. Converts each unique s3o -> a self-contained .glb (embedded WebP textures)
  5. Writes results to features/<name>/model.glb, dedup-ed by name
  6. Writes maps_features/<map_slug>/placements.json + heightmap.png so the
     map viewer can assemble a full scene (terrain + placed features).

Usage:
    python extract_map_features.py --map "altair crossing"
    python extract_map_features.py --map "eternal consequences"
"""

import argparse
import json
import math
import os
import re
import shutil
import struct
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import py7zr
from PIL import Image
from dotenv import load_dotenv

from feature_defs import (
    load_feature_dir,
    load_feature_file,
    get_object_path,
    get_normal_texture,
)
from set_lua_parser import parse_set_lua
from glb_feature_builder import build_feature_glb, TextureCache


# -- CONFIG ----------------------------------------------------------------

load_dotenv()

WEBFLOW_API_TOKEN = os.environ.get("WEBFLOW_API_TOKEN")
COLLECTION_ID = "6564c6553676389f8ba45aaf"
FIELD_DOWNLOAD_URL = "downloadurl"

HEADERS_WEBFLOW = {
    "Authorization": f"Bearer {WEBFLOW_API_TOKEN}",
    "accept-version": "2.0.0",
    "content-type": "application/json",
}

BAR_SDD_ROOT = r"C:\Games\Beyond-All-Reason\data\games\BAR.sdd"
BAR_FEATURES_DIR = os.path.join(BAR_SDD_ROOT, "features")
BAR_OBJECTS_DIR = os.path.join(BAR_SDD_ROOT, "objects3d")
BAR_UNITTEX_DIR = os.path.join(BAR_SDD_ROOT, "unittextures")

# Repo-relative output
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
FEATURES_OUT_DIR = os.path.join(REPO_ROOT, "features")
# Shared texture folder — all feature GLBs reference textures in here via
# relative URIs (../_textures/<name>.webp). Deduped across the whole run.
TEXTURES_OUT_DIR = os.path.join(FEATURES_OUT_DIR, "_textures")
# Per-map scene data (placements + heightmap) used by the map viewer.
MAPS_OUT_DIR = os.path.join(REPO_ROOT, "maps_features")

TEMP_ARCHIVE = "temp_features_map.sd7"
TEMP_EXTRACT = "temp_features_extract"


def _rmtree_with_retry(path: str, attempts: int = 4, delay: float = 0.5) -> None:
    """shutil.rmtree with a retry loop for transient Windows file locks
    (Explorer thumbnail cache, indexing service, desktop.ini handles).
    Silently no-ops if the path doesn't exist.
    """
    import stat
    import time
    if not os.path.exists(path):
        return

    def _onerror(func, p, exc_info):
        # Strip read-only attribute (common for desktop.ini) and retry once.
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except Exception:
            raise

    last_exc: Optional[BaseException] = None
    for i in range(attempts):
        try:
            shutil.rmtree(path, onerror=_onerror)
            return
        except Exception as e:
            last_exc = e
            if i < attempts - 1:
                time.sleep(delay)
    # Final attempt: best-effort with ignore_errors so we at least move on.
    shutil.rmtree(path, ignore_errors=True)
    if os.path.exists(path):
        # Something persistent is holding files — log but don't crash.
        print(f"   [!] Warning: could not fully clean {path}: {last_exc}")


# -- WEBFLOW LOOKUP --------------------------------------------------------

def fetch_webflow_map(map_filter: str) -> Optional[dict]:
    """Find the first Webflow map whose name contains map_filter (case-insens)."""
    url = f"https://api.webflow.com/v2/collections/{COLLECTION_ID}/items"
    offset = 0
    limit = 100
    while True:
        try:
            r = requests.get(url, headers=HEADERS_WEBFLOW,
                             params={'limit': limit, 'offset': offset})
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"[Webflow] API error: {e}")
            return None
        batch = data.get('items', [])
        if not batch:
            return None
        for item in batch:
            name = item.get('fieldData', {}).get('name', '')
            if map_filter.lower() in name.lower():
                return item
        if len(batch) < limit:
            return None
        offset += limit


# -- SD7 DOWNLOAD + EXTRACTION ---------------------------------------------

def download_sd7(url: str, dest: str):
    print(f"   -> Downloading: {url}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, 'wb') as f:
            shutil.copyfileobj(r.raw, f)


def _find_in_archive(names: List[str], needle: str) -> Optional[str]:
    """Case-insensitive suffix match for a path in an archive file list."""
    needle_low = needle.lower().replace('\\', '/')
    for n in names:
        if n.lower().replace('\\', '/').endswith(needle_low):
            return n
    return None


def extract_map_resources(sd7_path: str, extract_dir: str) -> dict:
    """Extract the subset of the sd7 we need: set.lua, features/*.lua,
    objects3d/**, unittextures/**, mapinfo.lua, maps/*.smf. Returns a
    resource index.
    """
    os.makedirs(extract_dir, exist_ok=True)
    with py7zr.SevenZipFile(sd7_path, mode='r') as z:
        all_names = z.getnames()

        wanted = []
        set_lua_name = None
        for n in all_names:
            low = n.lower().replace('\\', '/')
            if low.endswith('mapconfig/featureplacer/set.lua'):
                wanted.append(n)
                set_lua_name = n
            elif '/features/' in '/' + low and low.endswith('.lua'):
                wanted.append(n)
            elif low.startswith('features/') and low.endswith('.lua'):
                wanted.append(n)
            elif '/objects3d/' in '/' + low or low.startswith('objects3d/'):
                wanted.append(n)
            elif '/unittextures/' in '/' + low or low.startswith('unittextures/'):
                wanted.append(n)
            elif low == 'mapinfo.lua' or low.endswith('/mapinfo.lua'):
                # Skip 'maphelper/mapinfo.lua' — it's a stub, not the real one.
                if '/maphelper/' not in ('/' + low):
                    wanted.append(n)
            elif low.endswith('.smf'):
                wanted.append(n)

        if wanted:
            z.extract(targets=wanted, path=extract_dir)

    # Build resource index
    resources = {
        'set_lua_path': None,
        'features_dir': None,
        'objects3d_root': None,
        'unittextures_root': None,
        'mapinfo_lua_path': None,
        'smf_path': None,
    }

    for root, dirs, files in os.walk(extract_dir):
        low_root = root.lower().replace('\\', '/')
        if low_root.endswith('/mapconfig/featureplacer') and 'set.lua' in [f.lower() for f in files]:
            for f in files:
                if f.lower() == 'set.lua':
                    resources['set_lua_path'] = os.path.join(root, f)
        if low_root.endswith('/features') or low_root.endswith('\\features'):
            resources['features_dir'] = root
        if low_root.endswith('/objects3d') or low_root.endswith('\\objects3d'):
            resources['objects3d_root'] = root
        if low_root.endswith('/unittextures') or low_root.endswith('\\unittextures'):
            resources['unittextures_root'] = root
        for f in files:
            if f.lower() == 'mapinfo.lua' and 'maphelper' not in low_root:
                if resources['mapinfo_lua_path'] is None:
                    resources['mapinfo_lua_path'] = os.path.join(root, f)
            if f.lower().endswith('.smf'):
                if resources['smf_path'] is None:
                    resources['smf_path'] = os.path.join(root, f)

    return resources


# -- SMF / MAPINFO PARSING --------------------------------------------------

def parse_mapinfo_heights(mapinfo_path: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """Extract minheight / maxheight from a mapinfo.lua file.

    These are floats in Spring world units ("elmos") and define the
    real-world Y range that the SMF's uint16 heightmap samples cover.
    """
    if not mapinfo_path or not os.path.isfile(mapinfo_path):
        return None, None
    try:
        with open(mapinfo_path, 'r', encoding='utf-8', errors='replace') as f:
            src = f.read()
    except Exception:
        return None, None

    def _find(key: str) -> Optional[float]:
        m = re.search(rf'\b{key}\s*=\s*(-?[0-9]+(?:\.[0-9]+)?)', src, re.IGNORECASE)
        return float(m.group(1)) if m else None

    return _find('minheight'), _find('maxheight')


def parse_smf_heightmap(smf_path: str) -> Optional[dict]:
    """Parse a Spring .smf file and return its heightmap + world dimensions
    + the feature block (used by older maps to embed geovents).

    SMF header layout (first 80 bytes, little-endian):
        0..16   'spring map file\\0'
        16..20  version (uint32)
        20..24  mapid (uint32)
        24..28  mapx (uint32)  - map width in "squares"
        28..32  mapy (uint32)  - map height in "squares"
        32..36  squareSize (uint32) - world elmos per square (usually 8)
        36..40  texelPerSquare (uint32)
        40..44  tileSize (uint32)
        44..48  minHeight (float32) - world Y of heightmap value 0
        48..52  maxHeight (float32) - world Y of heightmap value 65535
        52..56  heightMapPtr (uint32)
        56..60  typeMapPtr (uint32)
        60..64  tilesPtr (uint32)
        64..68  miniMapPtr (uint32)
        68..72  metalMapPtr (uint32)
        72..76  featurePtr (uint32)
        76..80  numExtraHeaders (uint32)

    Heightmap is (mapx+1) × (mapy+1) uint16 values.
    World size = mapx * squareSize (elmos). For BAR that's mapx * 8.

    Feature block (at featurePtr):
        uint32 numFeatureType
        uint32 numFeatures
        numFeatureType × null-terminated ASCII names
        numFeatures × MapFeatureStruct {
            int32  featureType
            float  xpos, ypos, zpos
            float  rotation     (treated as int by Spring engine — heading 0..65536)
            float  relSize
        }   (24 bytes each)
    """
    try:
        with open(smf_path, 'rb') as f:
            header = f.read(80)
            if len(header) < 80 or header[:16].rstrip(b'\x00') != b'spring map file':
                return None
            mapx = struct.unpack('<I', header[24:28])[0]
            mapy = struct.unpack('<I', header[28:32])[0]
            square_size = struct.unpack('<I', header[32:36])[0]
            min_h = struct.unpack('<f', header[44:48])[0]
            max_h = struct.unpack('<f', header[48:52])[0]
            heightmap_offset = struct.unpack('<I', header[52:56])[0]
            feature_offset = struct.unpack('<I', header[72:76])[0]
            if mapx == 0 or mapy == 0 or square_size == 0:
                return None
            hm_w = mapx + 1
            hm_h = mapy + 1
            f.seek(heightmap_offset)
            buf = f.read(hm_w * hm_h * 2)
            if len(buf) < hm_w * hm_h * 2:
                return None
            arr = np.frombuffer(buf, dtype=np.uint16).reshape((hm_h, hm_w))

            # Feature block — only present on some maps. Older BAR maps put
            # geovents here instead of in set.lua, so we need to read them.
            smf_features: List[Dict[str, Any]] = []
            if feature_offset and feature_offset < os.path.getsize(smf_path):
                try:
                    f.seek(feature_offset)
                    n_types, n_feat = struct.unpack('<2I', f.read(8))
                    if 0 < n_types < 4096 and 0 <= n_feat < 1_000_000:
                        type_names: List[str] = []
                        for _ in range(n_types):
                            s = bytearray()
                            while True:
                                c = f.read(1)
                                if not c or c == b'\x00':
                                    break
                                s += c
                            type_names.append(s.decode('utf-8', 'replace'))
                        for _ in range(n_feat):
                            rec = f.read(24)
                            if len(rec) < 24:
                                break
                            ft = struct.unpack('<i', rec[0:4])[0]
                            xp, yp, zp = struct.unpack('<3f', rec[4:16])
                            # Rotation: stored as float in the file, but
                            # the Spring engine reinterprets it as a heading
                            # int (0..65536). Read as float — for geovents
                            # the value is normally 0 anyway.
                            rot = struct.unpack('<f', rec[16:20])[0]
                            if 0 <= ft < len(type_names):
                                smf_features.append({
                                    'name': type_names[ft].lower(),
                                    'x': xp,
                                    'y': yp,
                                    'z': zp,
                                    'rot': rot,
                                })
                except Exception as e:
                    print(f"   [!] SMF feature block parse failed: {e}")

            return {
                'mapx_squares': mapx,
                'mapy_squares': mapy,
                'square_size': square_size,
                'world_width': mapx * square_size,
                'world_height': mapy * square_size,
                'min_height_smf': float(min_h),
                'max_height_smf': float(max_h),
                'heightmap': arr,
                'smf_features': smf_features,
            }
    except Exception as e:
        print(f"   [!] SMF parse failed: {e}")
        return None


def sample_heightmap(hm: np.ndarray, x: float, z: float,
                     world_w: int, world_h: int,
                     min_h: float, max_h: float) -> float:
    """Sample a feature's Y from the heightmap at world (x,z).

    Uses bilinear interpolation. The heightmap grid is (h+1)×(w+1) samples
    over the world rectangle [0..world_w] × [0..world_h].
    """
    hm_h, hm_w = hm.shape
    # Normalise to grid coords
    gx = (x / world_w) * (hm_w - 1)
    gz = (z / world_h) * (hm_h - 1)
    gx = max(0.0, min(hm_w - 1.0001, gx))
    gz = max(0.0, min(hm_h - 1.0001, gz))
    x0, z0 = int(gx), int(gz)
    fx, fz = gx - x0, gz - z0
    v00 = hm[z0,     x0]
    v10 = hm[z0,     x0 + 1]
    v01 = hm[z0 + 1, x0]
    v11 = hm[z0 + 1, x0 + 1]
    v = ((1 - fx) * (1 - fz) * v00 +
         fx       * (1 - fz) * v10 +
         (1 - fx) * fz       * v01 +
         fx       * fz       * v11)
    # uint16 0..65535 → world Y in [min_h..max_h]
    return float(min_h + (v / 65535.0) * (max_h - min_h))


def write_heightmap_png(hm: np.ndarray, out_path: str) -> None:
    """Write the 16-bit heightmap to an 8-bit RGBA PNG, packing the high
    byte into R and the low byte into G. B/A are unused.

    Browsers downsample 16-bit grayscale PNGs to 8 bit when drawn to a
    canvas, which destroys the precision we care about. By packing the
    two bytes as separate color channels we preserve the full uint16
    value per sample, and the viewer recombines them as
    `hi * 256 + lo`, then remaps to [minHeight..maxHeight]."""
    hi = (hm >> 8).astype(np.uint8)
    lo = (hm & 0xFF).astype(np.uint8)
    h, w = hm.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = hi
    rgba[..., 1] = lo
    rgba[..., 3] = 255
    Image.fromarray(rgba, mode='RGBA').save(out_path, format='PNG', optimize=True)


# -- RESOURCE RESOLUTION ---------------------------------------------------

def _find_file_case_insensitive(search_root: str, rel_path: str) -> Optional[str]:
    """Given a folder and a relative path (e.g. 'rocks30/rocks30_def_01.s3o'),
    find the actual file on disk with case-insensitive matching.
    """
    if not search_root or not os.path.isdir(search_root):
        return None
    rel_norm = rel_path.replace('\\', '/').lstrip('/')
    target_low = rel_norm.lower()

    # Fast path: direct join exists
    candidate = os.path.join(search_root, rel_norm)
    if os.path.isfile(candidate):
        return candidate

    # Slow path: case-insensitive walk
    for root, dirs, files in os.walk(search_root):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), search_root)
            if rel.replace('\\', '/').lower() == target_low:
                return os.path.join(root, f)
    return None


def _find_file_by_basename(search_root: str, basename: str) -> Optional[str]:
    """Find any file whose basename matches (case-insens.) under search_root."""
    if not search_root or not os.path.isdir(search_root):
        return None
    target = basename.lower()
    for root, dirs, files in os.walk(search_root):
        for f in files:
            if f.lower() == target:
                return os.path.join(root, f)
    return None


def find_s3o(object_rel: str,
              map_objects3d: Optional[str],
              bar_objects3d: str) -> Optional[str]:
    """Locate an s3o file: first in the map's objects3d, then BAR's."""
    if not object_rel:
        return None
    # Spring stores object paths as e.g. 'rocks30/rocks30_def_01.s3o' or just
    # 'fir_tree_large.s3o'. We try case-insensitive relative lookup first, then
    # fall back to basename search.
    for root in (map_objects3d, bar_objects3d):
        if not root:
            continue
        hit = _find_file_case_insensitive(root, object_rel)
        if hit:
            return hit
        hit = _find_file_by_basename(root, os.path.basename(object_rel))
        if hit:
            return hit
    return None


def find_texture(tex_rel: str,
                  map_root: Optional[str],
                  map_unittex: Optional[str],
                  bar_unittex: str) -> Optional[str]:
    """Locate a texture file.

    tex_rel can be:
      - 'unittextures/rocks30_def_color.dds' (full relative path)
      - 'rocks30_def_color.dds' (just basename — look under unittextures)
      - stored inside the .s3o header as 'tex1.dds' (basename only)

    Also tries .dds / .png / .tga variants if the extension differs.
    """
    if not tex_rel:
        return None

    # Candidate paths to try (in priority order)
    candidates = [tex_rel]

    # If no subfolder, also try under unittextures/
    if '/' not in tex_rel and '\\' not in tex_rel:
        candidates.append(f"unittextures/{tex_rel}")
    else:
        # Full path — also try just the basename
        candidates.append(os.path.basename(tex_rel))

    # Try each extension variant (.dds / .png / .tga)
    base_no_ext, ext = os.path.splitext(candidates[0])
    alt_exts = ['.dds', '.png', '.tga']
    if ext.lower() in alt_exts:
        alt_exts.remove(ext.lower())
    for alt in alt_exts:
        candidates.append(base_no_ext + alt)
        candidates.append(os.path.basename(base_no_ext) + alt)

    search_roots = [
        # Map root first (in case the texture is at an unusual path)
        map_root,
        map_unittex,
        os.path.dirname(map_unittex) if map_unittex else None,
        bar_unittex,
        os.path.dirname(bar_unittex) if bar_unittex else None,
    ]

    for cand in candidates:
        for root in search_roots:
            if not root:
                continue
            hit = _find_file_case_insensitive(root, cand)
            if hit:
                return hit
            hit = _find_file_by_basename(root, os.path.basename(cand))
            if hit:
                return hit
    return None


def read_s3o_texture_names(s3o_path: str) -> Tuple[str, str]:
    """Quick read of tex1/tex2 names from an s3o file without full parse."""
    from s3o_parser import parse_s3o
    model = parse_s3o(s3o_path)
    return model.texture1, model.texture2


# -- MAIN PIPELINE ---------------------------------------------------------

def slugify_map_name(name: str) -> str:
    # Use hyphens — matches Webflow's CMS slug format so data-map-slug on
    # the live site lines up 1:1 with our maps_features/<slug>/ directory.
    s = re.sub(r'[^a-zA-Z0-9]+', '-', name.strip().lower())
    return s.strip('-')


def process_map(map_filter: str, force: bool = False):
    # 1. Look up map in Webflow
    print(f"\n[1/6] Looking up Webflow map matching \"{map_filter}\"...")
    item = fetch_webflow_map(map_filter)
    if not item:
        print(f"   [X] No Webflow map matching \"{map_filter}\"")
        return
    map_name = item.get('fieldData', {}).get('name', 'Unknown')
    map_slug = slugify_map_name(map_name)
    download_url = item.get('fieldData', {}).get(FIELD_DOWNLOAD_URL)
    if not download_url:
        print(f"   [X] Map \"{map_name}\" has no downloadurl field")
        return
    print(f"   [OK] Found: {map_name}  (slug: {map_slug})")

    # 2. Download + extract sd7
    print(f"\n[2/6] Downloading & extracting sd7...")
    # On Windows, Explorer's thumbnail cache / indexer can briefly lock
    # files (desktop.ini in particular) inside the temp extract. A simple
    # retry loop with a small delay clears these transient locks, while
    # still failing loudly if something more serious is going on.
    _rmtree_with_retry(TEMP_EXTRACT)
    if os.path.exists(TEMP_ARCHIVE):
        try:
            os.remove(TEMP_ARCHIVE)
        except OSError:
            pass

    try:
        download_sd7(download_url, TEMP_ARCHIVE)
        if not py7zr.is_7zfile(TEMP_ARCHIVE):
            print(f"   [X] Downloaded file is not a valid 7z archive")
            return
        resources = extract_map_resources(TEMP_ARCHIVE, TEMP_EXTRACT)
        print(f"   [OK] set.lua:         {resources['set_lua_path']}")
        print(f"   [OK] features dir:    {resources['features_dir']}")
        print(f"   [OK] objects3d:       {resources['objects3d_root']}")
        print(f"   [OK] unittextures:    {resources['unittextures_root']}")

        if not resources['set_lua_path']:
            print(f"   [X] No set.lua found in map — nothing to do")
            return

        # 3. Parse set.lua placements
        print(f"\n[3/6] Parsing set.lua...")
        with open(resources['set_lua_path'], 'r', encoding='utf-8', errors='replace') as f:
            set_source = f.read()
        placements = parse_set_lua(set_source)
        print(f"   [OK] {len(placements)} placements from set.lua")

        # Parse SMF early so we can merge any geovents that older maps embed
        # in the SMF binary instead of set.lua. We reuse smf_info later for
        # the heightmap step too.
        smf_info = None
        if resources['smf_path']:
            smf_info = parse_smf_heightmap(resources['smf_path'])

        if smf_info and smf_info.get('smf_features'):
            existing_geovent = any(p['name'] == 'geovent' for p in placements)
            if not existing_geovent:
                added = 0
                for sf in smf_info['smf_features']:
                    if sf['name'] == 'geovent':
                        placements.append({
                            'category': 'smf',
                            'name': 'geovent',
                            'x': sf['x'],
                            'z': sf['z'],
                            'rot': sf['rot'],
                        })
                        added += 1
                if added:
                    print(f"   [OK] +{added} geovents from SMF binary")

        print(f"   [OK] {len(placements)} placements total")
        unique_names = sorted(set(p['name'] for p in placements))
        print(f"   [OK] {len(unique_names)} unique feature names")

        # 4. Load FeatureDefs (map-local first, then BAR.sdd)
        print(f"\n[4/6] Loading FeatureDefs...")
        map_defs: Dict[str, dict] = {}
        if resources['features_dir']:
            map_defs = load_feature_dir(resources['features_dir'])
            print(f"   [OK] {len(map_defs)} map-local FeatureDefs")
        bar_defs = load_feature_dir(BAR_FEATURES_DIR)
        print(f"   [OK] {len(bar_defs)} BAR.sdd FeatureDefs")

        # 5. Resolve + convert each unique feature
        print(f"\n[5/6] Resolving & converting features...")

        os.makedirs(FEATURES_OUT_DIR, exist_ok=True)

        # One shared texture cache for the whole run — every GLB references
        # textures in FEATURES_OUT_DIR/_textures via a relative URI, so
        # identical sources are written to disk exactly once.
        texture_cache = TextureCache(TEXTURES_OUT_DIR, map_name=map_name)

        stats = {
            'resolved_map': 0,
            'resolved_bar': 0,
            'unresolved': 0,
            'converted': 0,
            'skipped_existing': 0,
            'conversion_failed': 0,
        }
        unresolved_names = []

        for name in unique_names:
            # Geovents have no model — they're rendered client-side as a
            # smoke particle effect by the viewer/embed, but the placement
            # is still emitted to placements.json so the embed knows where
            # to spawn the smoke.
            if name == 'geovent':
                continue

            out_dir = os.path.join(FEATURES_OUT_DIR, name)
            out_glb = os.path.join(out_dir, 'model.glb')

            if os.path.isfile(out_glb) and not force:
                stats['skipped_existing'] += 1
                continue

            # Resolve FeatureDef
            fdef = map_defs.get(name)
            resolved_from = None
            if fdef:
                resolved_from = 'map'
            else:
                fdef = bar_defs.get(name)
                if fdef:
                    resolved_from = 'bar'

            if not fdef:
                stats['unresolved'] += 1
                unresolved_names.append(name)
                continue

            stats[f'resolved_{resolved_from}'] += 1

            # Locate s3o
            object_rel = get_object_path(fdef)
            if not object_rel:
                print(f"   [X] {name}: FeatureDef has no 'object' field")
                stats['conversion_failed'] += 1
                continue

            s3o_path = find_s3o(
                object_rel,
                resources['objects3d_root'],
                BAR_OBJECTS_DIR,
            )
            if not s3o_path:
                print(f"   [X] {name}: s3o not found ({object_rel})")
                stats['conversion_failed'] += 1
                continue

            # Locate textures
            try:
                tex1, tex2 = read_s3o_texture_names(s3o_path)
            except Exception as e:
                print(f"   [X] {name}: s3o parse failed: {e}")
                stats['conversion_failed'] += 1
                continue

            color_path = find_texture(
                tex1,
                map_root=TEMP_EXTRACT,
                map_unittex=resources['unittextures_root'],
                bar_unittex=BAR_UNITTEX_DIR,
            )
            # Normal map comes ONLY from customparams.normaltex. BAR's tex2
            # slot is the "Extra" texture (spec/emit/mask), NOT a normal map
            # — using it as such produces garbage normals.
            normal_rel = get_normal_texture(fdef)
            normal_path = find_texture(
                normal_rel,
                map_root=TEMP_EXTRACT,
                map_unittex=resources['unittextures_root'],
                bar_unittex=BAR_UNITTEX_DIR,
            ) if normal_rel else None

            # The Extra texture (tex2): its alpha channel is the feature
            # mask for transparency (leaves, etc.). We pass it to the builder
            # so it can splice that alpha onto the color texture when the
            # color's own alpha is empty.
            extra_path = find_texture(
                tex2,
                map_root=TEMP_EXTRACT,
                map_unittex=resources['unittextures_root'],
                bar_unittex=BAR_UNITTEX_DIR,
            ) if tex2 else None

            # Build GLB — ensure the target directory exists *before* building,
            # so the image URI can be computed relative to that directory.
            os.makedirs(out_dir, exist_ok=True)
            glb_bytes = build_feature_glb(
                s3o_path, color_path, normal_path,
                texture_cache=texture_cache,
                glb_out_dir=out_dir,
                extra_tex_path=extra_path,
            )
            if glb_bytes is None:
                stats['conversion_failed'] += 1
                continue

            with open(out_glb, 'wb') as f:
                f.write(glb_bytes)

            size_kb = len(glb_bytes) / 1024
            tex_info = []
            if color_path:
                tex_info.append(f"color={os.path.basename(color_path)}")
            if normal_path:
                tex_info.append(f"normal={os.path.basename(normal_path)}")
            tex_str = f"  [{', '.join(tex_info)}]" if tex_info else ""
            print(f"   [OK] {name}  ({resolved_from}, {size_kb:.0f} KB){tex_str}")
            stats['converted'] += 1

        # 6. Write per-map scene data: placements.json + heightmap.png
        # smf_info was already parsed in step 3 (so we could merge any
        # geovents embedded in the SMF binary into the placement list).
        print(f"\n[6/6] Writing scene data...")
        map_out_dir = os.path.join(MAPS_OUT_DIR, map_slug)
        os.makedirs(map_out_dir, exist_ok=True)

        if smf_info:
            print(f"   [OK] SMF: {smf_info['world_width']}x{smf_info['world_height']} elmos "
                  f"(mapx={smf_info['mapx_squares']} squares of {smf_info['square_size']}), "
                  f"heightmap {smf_info['heightmap'].shape[1]}x{smf_info['heightmap'].shape[0]}")
        elif resources['smf_path']:
            print(f"   [!] SMF parse failed")
        else:
            print(f"   [!] No SMF file found in sd7")

        # Prefer the SMF header's minHeight/maxHeight — those are the values
        # the engine actually uses to decode the uint16 heightmap. mapinfo.lua
        # sometimes lists different values for gameplay purposes (e.g. max
        # build height), so only use it as a fallback.
        min_h = max_h = None
        if smf_info is not None:
            min_h = smf_info['min_height_smf']
            max_h = smf_info['max_height_smf']
        if min_h is None or max_h is None:
            lm, lM = parse_mapinfo_heights(resources['mapinfo_lua_path'])
            min_h = lm if lm is not None else 0.0
            max_h = lM if lM is not None else 1000.0
        print(f"   [OK] Height range: {min_h} .. {max_h}")

        # Heightmap PNG (16-bit grayscale, full precision)
        heightmap_rel = None
        if smf_info is not None:
            heightmap_path = os.path.join(map_out_dir, 'heightmap.png')
            write_heightmap_png(smf_info['heightmap'], heightmap_path)
            heightmap_rel = 'heightmap.png'
            print(f"   [OK] Heightmap -> {heightmap_path}")

        # Enrich placements with y sampled from heightmap, and convert
        # rotation from Spring heading (int16, 0..65535 = 2*pi) to radians.
        SPRING_HEADING_TO_RAD = (2.0 * math.pi) / 65536.0

        enriched = []
        if smf_info is not None:
            world_w = smf_info['world_width']
            world_h = smf_info['world_height']
            hm = smf_info['heightmap']
            for p in placements:
                y = sample_heightmap(hm, p['x'], p['z'], world_w, world_h, min_h, max_h)
                enriched.append({
                    'name': p['name'],
                    'x': p['x'],
                    'y': y,
                    'z': p['z'],
                    'rotY': p['rot'] * SPRING_HEADING_TO_RAD,
                })
        else:
            # No heightmap — y defaults to 0
            for p in placements:
                enriched.append({
                    'name': p['name'],
                    'x': p['x'],
                    'y': 0.0,
                    'z': p['z'],
                    'rotY': p['rot'] * SPRING_HEADING_TO_RAD,
                })

        scene = {
            'name': map_name,
            'slug': map_slug,
            'worldWidth': smf_info['world_width'] if smf_info else None,
            'worldHeight': smf_info['world_height'] if smf_info else None,
            'minHeight': min_h,
            'maxHeight': max_h,
            'heightmap': heightmap_rel,
            'featuresBase': '../features',
            'features': enriched,
        }
        scene_path = os.path.join(map_out_dir, 'placements.json')
        with open(scene_path, 'w', encoding='utf-8') as f:
            json.dump(scene, f, indent=1)
        print(f"   [OK] Placements ({len(enriched)}) -> {scene_path}")

        # Summary
        print(f"\n--- Summary for \"{map_name}\" ---")
        print(f"  Unique features: {len(unique_names)}")
        print(f"  Resolved from map:  {stats['resolved_map']}")
        print(f"  Resolved from BAR:  {stats['resolved_bar']}")
        print(f"  Unresolved:         {stats['unresolved']}")
        print(f"  New GLB written:    {stats['converted']}")
        print(f"  Skipped (existing): {stats['skipped_existing']}")
        print(f"  Conversion failed:  {stats['conversion_failed']}")
        tc = texture_cache.stats
        print(f"  Textures unique:    {tc['unique_files']}")
        print(f"  Textures reused:    {tc['reused_by_source']} (by source) + "
              f"{tc['reused_by_content']} (by content) + "
              f"{tc.get('reused_by_perceptual', 0)} (perceptual)")
        print(f"  Texture variants:   {tc.get('variant_saved', 0)} (same name, different content)")
        print(f"  Texture total size: {tc['total_bytes']/1024:.0f} KB")
        if unresolved_names:
            print(f"\n  Unresolved names (first 20):")
            for n in unresolved_names[:20]:
                print(f"    - {n}")

    finally:
        # By default we keep the extracted temp dir around so debugging a
        # map doesn't require re-downloading it. Set CLEAN_TEMP=1 to wipe
        # it on exit (e.g. in bulk rebuilds where disk space matters).
        if os.environ.get('CLEAN_TEMP'):
            if os.path.exists(TEMP_EXTRACT):
                shutil.rmtree(TEMP_EXTRACT, ignore_errors=True)
            if os.path.exists(TEMP_ARCHIVE):
                os.remove(TEMP_ARCHIVE)


def main():
    parser = argparse.ArgumentParser(
        description="Extract map feature models + textures to GLB files")
    parser.add_argument('--map', type=str, required=True,
                        help='Map name to process (case-insensitive partial match)')
    parser.add_argument('--force', action='store_true',
                        help='Re-convert features even if GLB already exists')
    args = parser.parse_args()

    if not WEBFLOW_API_TOKEN:
        print("ERROR: WEBFLOW_API_TOKEN not set in .env")
        sys.exit(1)

    if not os.path.isdir(BAR_SDD_ROOT):
        print(f"ERROR: BAR.sdd not found at {BAR_SDD_ROOT}")
        sys.exit(1)

    process_map(args.map, force=args.force)


if __name__ == "__main__":
    main()
