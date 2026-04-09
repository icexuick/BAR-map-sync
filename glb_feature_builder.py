"""
GLB Feature Builder — converts a parsed S3OModel + texture files into a
.glb referencing externally-stored, deduped WebP textures.

Output layout:
    features/
      _textures/
        ad0_fir_1.webp            # shared across every feature that uses it
        ad0_fir_normal.webp
        ...
      fir_tree_tall_5__tree_fir_tall_1/
        model.glb                 # image.uri = "../_textures/ad0_fir_1.webp"
      ...

Key design points:
  - TextureCache is constructed once per run and shared across all builds.
    It dedupes by (source_basename, content hash) so identical sources yield
    a single file on disk and every referring GLB points at that same URI.
  - Naming is human-readable: the source basename (e.g. "ad0_fir_1.webp").
    If two different source files share a basename but have different pixel
    content, the second one is disambiguated with a short content-hash
    suffix ("ad0_fir_1__a3f2c91b.webp").
  - glTF image.uri is relative from the GLB file location, i.e. for a GLB
    at features/<name>/model.glb the URI is "../_textures/<file>.webp".
  - No EXT_texture_webp extension is declared — glTF 2.0 natively supports
    image/webp when delivered via URI, and every modern loader handles it.
  - Tangent vectors are generated for meshes whose material has a normal
    texture, eliminating the MESH_PRIMITIVE_GENERATED_TANGENT_SPACE warning.
"""

import hashlib
import io
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import imageio.v3 as iio
from PIL import Image

# Make example-scripts importable
_EXAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'example-scripts')
if _EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLE_DIR)

from s3o_parser import parse_s3o, S3OModel  # noqa: E402
from s3o_to_glb import GLBBuilder  # noqa: E402


# Texture downscaling policy for feature textures:
#   - Halve each source dimension exactly once
#   - Clamp the result so no dimension drops below MIN_TEXTURE_DIMENSION
#   - If the source is already smaller than MIN_TEXTURE_DIMENSION, keep it
#     untouched (e.g. a 128x128 source stays 128x128)
MIN_TEXTURE_DIMENSION = 256

# WebP quality for color textures (perceptual lossy is fine)
COLOR_WEBP_QUALITY = 80


def _load_raw_rgba(path: str) -> Optional[Image.Image]:
    """Load an image with its alpha channel intact — no all-zero-alpha
    demotion. Used when we want to inspect the alpha channel of BAR's
    "Extra"/tex2 texture, whose alpha often carries the feature mask.
    Returns a Pillow RGBA image, or None on any error.
    """
    if not path or not os.path.isfile(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in ('.dds', '.tga'):
            arr = iio.imread(path)
            if arr.ndim == 2:
                return Image.fromarray(arr, mode='L').convert('RGBA')
            if arr.shape[2] == 3:
                return Image.fromarray(arr, mode='RGB').convert('RGBA')
            if arr.shape[2] == 4:
                return Image.fromarray(arr, mode='RGBA')
            return None
        img = Image.open(path)
        img.load()
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        return img
    except Exception as e:
        print(f"      [Texture] Raw load error {os.path.basename(path)}: {e}")
        return None


def _derive_mask_from_extra(extra_rgba: Image.Image) -> Optional[Image.Image]:
    """Given a raw RGBA of BAR's tex2/"Extra" texture, derive a single-channel
    alpha-mask Image ('L' mode). Returns None if the Extra texture doesn't
    carry a mask.

    BAR tex2 layout:
      - RGB = PBR-ish channels (roughness, metalness, emission, team color)
      - A   = alpha-mask for the model (when present)

    We use the alpha channel directly. A varying alpha channel (not constant
    0 and not constant 255) is treated as the mask. RGB is ignored for
    mask derivation — we previously had a "RGB=(0,0,0) = holes" heuristic
    for "treeshader" atlases but that was wrong: in BAR RGB in tex2 is
    PBR data, not a mask, and low/zero values just mean low roughness etc.
    """
    if extra_rgba.mode != 'RGBA':
        return None
    a = extra_rgba.split()[3]
    a_lo, a_hi = a.getextrema()
    # Alpha is a usable mask if it has non-trivial variation.
    if a_hi > 0 and a_hi - a_lo > 0:
        return a
    return None


def _load_image_any(path: str) -> Optional[Image.Image]:
    """Load a .dds / .png / .tga / .jpg into a Pillow image.

    Returns an RGBA image, with the exception that if the source's alpha
    channel has no variation (constant value — e.g. all-zero, all-255, or
    a stray constant like all-3), we return an RGB image instead so the
    material logic treats it as opaque. A flat alpha conveys no mask info
    and — in the all-near-zero case — would make the feature invisible if
    kept and fed to alphaMode=MASK.

    We use imageio for .dds and .tga because Pillow's TGA loader mishandles
    some BAR assets (e.g. pilha_crystal_teal_tex1.tga reads back as pure
    black via Pillow due to an alpha-premultiplication quirk).
    """
    if not path or not os.path.isfile(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in ('.dds', '.tga'):
            arr = iio.imread(path)
            if arr.ndim == 2:
                img = Image.fromarray(arr, mode='L').convert('RGB')
            elif arr.shape[2] == 3:
                img = Image.fromarray(arr, mode='RGB')
            elif arr.shape[2] == 4:
                alpha_min = int(arr[:, :, 3].min())
                alpha_max = int(arr[:, :, 3].max())
                if alpha_min == alpha_max:
                    # Constant alpha (any value) carries no mask info; drop it.
                    img = Image.fromarray(arr[:, :, :3], mode='RGB')
                else:
                    img = Image.fromarray(arr, mode='RGBA')
            else:
                return None
        else:
            img = Image.open(path)
            img.load()
            if img.mode == 'RGBA':
                a_lo, a_hi = img.split()[-1].getextrema()
                if a_lo == a_hi:
                    img = img.convert('RGB')
            elif img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
    except Exception as e:
        print(f"      [Texture] Load error {os.path.basename(path)}: {e}")
        return None
    return img


def _halve_once_clamped(width: int, height: int) -> Tuple[int, int]:
    """Halve-once, clamp-to-256 policy. See module docstring."""
    src_max = max(width, height)
    if src_max <= MIN_TEXTURE_DIMENSION:
        return width, height

    half_w = max(1, width // 2)
    half_h = max(1, height // 2)
    halved_max = max(half_w, half_h)

    if halved_max >= MIN_TEXTURE_DIMENSION:
        return half_w, half_h

    scale = MIN_TEXTURE_DIMENSION / halved_max
    return max(1, round(half_w * scale)), max(1, round(half_h * scale))


def _encode_webp(img: Image.Image, lossless: bool, keep_full_res: bool = False) -> bytes:
    """Resize (per halve-once policy) and encode a Pillow image to WebP bytes.

    keep_full_res: skip the halve-once downscale entirely. Used for assets
        (e.g. rocks30) where the extra texel detail is needed to inspect
        the feature in the viewer without the atlas blurring into mush.
    """
    if keep_full_res:
        new_w, new_h = img.width, img.height
    else:
        new_w, new_h = _halve_once_clamped(img.width, img.height)
    if (new_w, new_h) != (img.width, img.height):
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    if lossless:
        img.save(buf, format='WEBP', lossless=True, method=6)
    else:
        img.save(buf, format='WEBP', quality=COLOR_WEBP_QUALITY, method=6)
    return buf.getvalue()


def _safe_basename(source_path: str) -> str:
    """Return a lowercase, extension-less basename safe for a filesystem."""
    base = os.path.splitext(os.path.basename(source_path))[0].lower()
    # Replace anything that's not alnum/_/- with _
    out = []
    for ch in base:
        if ch.isalnum() or ch in ('_', '-'):
            out.append(ch)
        else:
            out.append('_')
    return ''.join(out) or 'texture'


# ──────────────────────────────────────────────────────────────────────────
# Perceptual texture comparison
# ──────────────────────────────────────────────────────────────────────────

# Thresholds for deciding whether two same-named textures are meaningfully
# different. Both conditions must be met for a variant to be saved.
#
# Derived empirically from comparing greenest-fields source TGAs against
# previously-cached textures of the same basename:
#   - ad0_aleppo2_1:  8% pixel-diff but 0.8% mean color delta  → compression noise
#   - shroomorange:  31% pixel-diff, 30% mean color delta      → clearly different
#   - fernsa:        16% pixel-diff,  8-12% mean color delta   → clearly different
TEXTURE_DIFF_PIXEL_FRAC = 0.05     # >5% of pixels must visibly differ
TEXTURE_DIFF_MEAN_COLOR = 0.03     # AND mean color delta >3% (~7.65/255)
# "Visibly differ" per pixel means any channel moved by more than this:
TEXTURE_DIFF_PIXEL_DELTA = 12      # ~5% of 255


def _compare_images_rgba(a: Image.Image, b: Image.Image) -> Tuple[float, float]:
    """Compare two decoded images and return (pixel_frac, mean_color_delta).

    pixel_frac:        fraction of pixels where any channel differs by more
                       than TEXTURE_DIFF_PIXEL_DELTA (0..1)
    mean_color_delta:  mean absolute per-channel delta, normalized to 0..1

    Both images are resized to a common small footprint (128x128) before
    comparison so that encoding-level size differences don't dominate. RGBA
    with zero-alpha regions is kept as-is (we want to detect mask changes
    too, since alpha is meaningful for features like leaves).
    """
    SIZE = (128, 128)
    ai = a.convert('RGBA').resize(SIZE, Image.Resampling.LANCZOS)
    bi = b.convert('RGBA').resize(SIZE, Image.Resampling.LANCZOS)
    aa = np.asarray(ai, dtype=np.int16)
    bb = np.asarray(bi, dtype=np.int16)
    diff = np.abs(aa - bb)
    # per-pixel max channel delta
    max_chan = diff.max(axis=2)
    pixel_frac = float((max_chan > TEXTURE_DIFF_PIXEL_DELTA).mean())
    mean_color = float(diff.mean()) / 255.0
    return pixel_frac, mean_color


def _is_meaningfully_different(a: Image.Image, b: Image.Image) -> bool:
    """True if two textures should be stored as separate variants."""
    pixel_frac, mean_color = _compare_images_rgba(a, b)
    return pixel_frac > TEXTURE_DIFF_PIXEL_FRAC and mean_color > TEXTURE_DIFF_MEAN_COLOR


# ──────────────────────────────────────────────────────────────────────────
# TextureCache: shared, deduped, human-readable texture store on disk
# ──────────────────────────────────────────────────────────────────────────

class TextureCache:
    """Writes WebP textures to a shared folder, deduped by content hash.

    Usage:
        cache = TextureCache(textures_dir="features/_textures")
        filename = cache.register(source_path="...tex1.dds", lossless=False)
        # => "ad0_fir_1.webp"

    Naming policy:
      - Preferred name is the source basename with .webp extension.
      - If that name is already taken by *different* content, a short
        content-hash suffix is appended ("ad0_fir_1__a3f2c91b.webp").
      - Two calls with identical encoded bytes always return the same
        filename (pure content-hash dedup).

    Not thread-safe. One instance per run.
    """

    def __init__(self, textures_dir: str, map_name: Optional[str] = None):
        self.textures_dir = textures_dir
        self.map_name = map_name
        os.makedirs(textures_dir, exist_ok=True)

        # Maps: source_path (absolute, lowercased) -> filename
        self._by_source: Dict[str, str] = {}
        # Maps: content_sha1 -> filename (pure content dedup)
        self._by_hash: Dict[str, str] = {}
        # Names currently in use on disk (lowercased filenames) -> content_sha1
        self._name_to_hash: Dict[str, str] = {}
        # Stem (no extension) -> list of variant filenames that share this stem.
        # Used to find candidates for perceptual comparison on collision.
        # Example: 'shroomorange' -> ['shroomorange.webp', 'shroomorange__greenest_fields.webp']
        self._variants_by_stem: Dict[str, list] = {}
        # Lazy-decoded pixel cache: filename -> Pillow image (or None if unreadable)
        self._decoded_by_filename: Dict[str, Optional[Image.Image]] = {}

        self.stats = {
            'unique_files': 0,
            'reused_by_source': 0,
            'reused_by_content': 0,
            'reused_by_perceptual': 0,
            'variant_saved': 0,
            'name_collisions': 0,
            'total_bytes': 0,
        }

        # Bootstrap from any files already on disk, so cross-map dedup works
        # across independent cache instances (one per map run).
        self._bootstrap_from_disk()

    def _bootstrap_from_disk(self) -> None:
        """Index every *.webp already in textures_dir so this run can dedupe
        against textures written by earlier runs (e.g. previous maps).
        """
        if not os.path.isdir(self.textures_dir):
            return
        for name in os.listdir(self.textures_dir):
            if not name.lower().endswith('.webp'):
                continue
            path = os.path.join(self.textures_dir, name)
            if not os.path.isfile(path):
                continue
            try:
                with open(path, 'rb') as f:
                    data = f.read()
            except OSError:
                continue
            content_hash = hashlib.sha1(data).hexdigest()
            self._by_hash.setdefault(content_hash, name)
            self._name_to_hash[name.lower()] = content_hash
            stem = self._stem_of_variant(name)
            self._variants_by_stem.setdefault(stem, []).append(name)

    @staticmethod
    def _stem_of_variant(filename: str) -> str:
        """Extract the logical stem from a variant filename.
        'shroomorange.webp'                    -> 'shroomorange'
        'shroomorange__greenest_fields.webp'   -> 'shroomorange'
        'ad0_fir_1_masked.webp'                -> 'ad0_fir_1_masked'
        'ad0_fir_1__a3f2c91b.webp'             -> 'ad0_fir_1'
        The '__' separator is the variant boundary; whatever comes after it
        (mapname or content hash) is treated as a variant suffix.
        """
        base = os.path.splitext(filename)[0].lower()
        idx = base.rfind('__')
        if idx >= 0:
            return base[:idx]
        return base

    def _get_decoded(self, filename: str) -> Optional[Image.Image]:
        """Lazily decode and cache a webp from disk for perceptual comparison."""
        if filename in self._decoded_by_filename:
            return self._decoded_by_filename[filename]
        path = os.path.join(self.textures_dir, filename)
        img: Optional[Image.Image] = None
        try:
            im = Image.open(path)
            im.load()
            img = im
        except Exception:
            img = None
        self._decoded_by_filename[filename] = img
        return img

    def register(self, source_path: str, lossless: bool,
                 match_size_of: Optional[str] = None,
                 keep_full_res: bool = False) -> Optional[str]:
        """Encode + store the texture, returning the filename (no dir prefix).
        Returns None if the source can't be loaded.

        match_size_of: optional path to another image whose dimensions this
            texture must be downscaled to match before encoding. Used to
            keep a normal map at the same resolution as its paired color
            texture so tangent-space lookups sample at the same per-UV-texel
            rate — otherwise a high-res normal over a tiny UV region
            produces "random dark blobs" that don't correspond to diffuse
            detail (see rocks30, whose normal DDS is 2× the color DDS).
            Only shrinks, never upscales.
        """
        if not source_path:
            return None
        key = os.path.normcase(os.path.abspath(source_path))
        # Include match-size target and full-res flag in the dedup key so
        # the same source can coexist in halved and full-res variants.
        if match_size_of:
            key = key + '|match=' + os.path.normcase(os.path.abspath(match_size_of))
        if keep_full_res:
            key = key + '|full'
        cached = self._by_source.get(key)
        if cached:
            self.stats['reused_by_source'] += 1
            return cached

        img = _load_image_any(source_path)
        if img is None:
            return None

        if match_size_of:
            partner = _load_image_any(match_size_of)
            if partner is not None and partner.size != img.size:
                tw, th = partner.size
                # Only shrink, never upscale.
                if tw * th < img.width * img.height:
                    img = img.resize((tw, th), Image.Resampling.LANCZOS)

        return self._store_image(img, source_path, lossless, key, keep_full_res=keep_full_res)

    def register_color_with_mask(self,
                                  color_path: str,
                                  mask_source_path: Optional[str],
                                  keep_full_res: bool = False) -> Tuple[Optional[str], bool]:
        """Register a color texture, optionally borrowing its alpha channel
        from another image (BAR's tex2/"Extra") when the color's own alpha
        is unusable.

        BAR feature convention:
          - tex1 (color): RGB=diffuse, A=often all-zero/unused
          - tex2 (extra): RGB=specular/emit/team, A=alpha MASK for the model

        Our material expects the alpha-mask inside the baseColorTexture
        (glTF's alphaMode=MASK reads material.baseColorTexture.A). So when
        tex1 has no useful alpha, we splice tex2's alpha onto tex1 here.

        Returns (filename, has_mask):
          filename: the cached webp to reference from baseColorTexture
          has_mask: True if an alpha mask (from tex1 OR tex2) is now present
                    and the material should be marked alphaMode=MASK
        """
        if not color_path:
            return None, False

        color_img = _load_image_any(color_path)
        if color_img is None:
            return None, False

        # Does the color already carry a usable alpha channel? _load_image_any
        # has already demoted all-zero-alpha RGBA to RGB, so 'A' in bands
        # implies useful alpha.
        color_has_alpha = 'A' in color_img.getbands()

        # Can we get a better mask from the extra/tex2 texture?
        # BAR's tex2 "Extra" carries the feature mask in one of two shapes:
        #   (a) In the alpha channel directly (some ad0_* trees, crystals)
        #   (b) Encoded in the RGB itself — pixels where RGB==(0,0,0) are
        #       the alpha-mask holes, non-zero pixels are opaque. BAR's
        #       treeshader uses this layout (e.g. gasbag_tree, allpinesb).
        merged = False
        if not color_has_alpha and mask_source_path:
            raw = _load_raw_rgba(mask_source_path)
            if raw is not None:
                mask = _derive_mask_from_extra(raw)
                if mask is not None:
                    # Resize to match color texture dimensions if needed
                    if mask.size != color_img.size:
                        mask = mask.resize(color_img.size, Image.Resampling.LANCZOS)
                    rgb = color_img.convert('RGB')
                    r, g, b = rgb.split()
                    color_img = Image.merge('RGBA', (r, g, b, mask))
                    merged = True

        has_mask = color_has_alpha or merged

        # Build a deterministic source-key so repeat calls with the same
        # (color, mask) pair dedupe via _by_source.
        key_parts = [os.path.normcase(os.path.abspath(color_path))]
        if merged:
            key_parts.append(os.path.normcase(os.path.abspath(mask_source_path)))
        key = '|masked|'.join(key_parts)
        if keep_full_res:
            key = key + '|full'

        cached = self._by_source.get(key)
        if cached:
            self.stats['reused_by_source'] += 1
            return cached, has_mask

        filename = self._store_image(
            color_img, color_path, lossless=False, source_key=key,
            name_suffix='_masked' if merged else '',
            keep_full_res=keep_full_res,
        )
        return filename, has_mask

    def _store_image(self,
                      img: Image.Image,
                      source_path_for_name: str,
                      lossless: bool,
                      source_key: str,
                      name_suffix: str = '',
                      keep_full_res: bool = False) -> Optional[str]:
        """Encode `img` to webp, dedup by content hash, write to disk, and
        record both the source-key mapping and the content-hash mapping.
        Returns the filename on disk (no directory prefix).

        Collision policy (when a file with the preferred name already exists
        on disk but its bytes differ):
          1. Decode the existing file and run a perceptual comparison.
          2. If the two images are visually close (compression noise only),
             reuse the existing file — save nothing new.
          3. If they're meaningfully different, save the new image under
             '<stem>__<mapname>.webp' (or '<stem>__<hash8>.webp' as a
             fallback when no map name is known).
        """
        data = _encode_webp(img, lossless=lossless, keep_full_res=keep_full_res)
        content_hash = hashlib.sha1(data).hexdigest()

        # Content-hash dedup: if we've already written these exact bytes, reuse.
        existing = self._by_hash.get(content_hash)
        if existing:
            self._by_source[source_key] = existing
            self.stats['reused_by_content'] += 1
            return existing

        # Pick a human-readable filename, disambiguating on collision.
        stem = _safe_basename(source_path_for_name) + name_suffix
        preferred = stem + '.webp'
        logical_stem = self._stem_of_variant(preferred)

        # Perceptual comparison against any existing variant(s) of this stem.
        # This catches two cases:
        #   (a) same-source re-encoded in a later run → compression noise only,
        #       reuse the existing file instead of proliferating variants.
        #   (b) a truly different texture with the same name → save a variant.
        # We compare the post-encoding round-trip (decoded from `data`)
        # against the on-disk decoded variant, so both sides have gone
        # through WebP compression and neither has a size advantage.
        existing_variants = self._variants_by_stem.get(logical_stem, [])
        if existing_variants:
            try:
                new_decoded = Image.open(io.BytesIO(data))
                new_decoded.load()
            except Exception:
                new_decoded = img  # fallback: use pre-encoding image
            for variant_name in existing_variants:
                decoded = self._get_decoded(variant_name)
                if decoded is None:
                    continue
                if not _is_meaningfully_different(new_decoded, decoded):
                    # Visually the same — point at the existing file.
                    self._by_source[source_key] = variant_name
                    # Also index the new content hash → existing filename so
                    # future byte-identical encodes shortcut via _by_hash.
                    self._by_hash[content_hash] = variant_name
                    self.stats['reused_by_perceptual'] += 1
                    return variant_name

            # No variant matched — this is a genuinely different image.
            # Pick a variant suffix: mapname if known, else short content hash.
            if self.map_name:
                suffix = _safe_basename(self.map_name)
            else:
                suffix = content_hash[:8]
            filename = f"{stem}__{suffix}.webp"
            # Defensive: if that variant name is somehow already taken by
            # different content, fall back to a hash suffix.
            if filename.lower() in self._name_to_hash and self._name_to_hash[filename.lower()] != content_hash:
                filename = f"{stem}__{content_hash[:8]}.webp"
            self.stats['variant_saved'] += 1
            self.stats['name_collisions'] += 1
        else:
            filename = preferred

        out_path = os.path.join(self.textures_dir, filename)
        with open(out_path, 'wb') as f:
            f.write(data)

        self._by_source[source_key] = filename
        self._by_hash[content_hash] = filename
        self._name_to_hash[filename.lower()] = content_hash
        self._variants_by_stem.setdefault(logical_stem, []).append(filename)
        # Cache the decoded pixels for fast comparison on subsequent collisions.
        self._decoded_by_filename[filename] = img
        self.stats['unique_files'] += 1
        self.stats['total_bytes'] += len(data)
        return filename


# ──────────────────────────────────────────────────────────────────────────
# FeatureGLBBuilder: extends parent with textured material + tangent support
# ──────────────────────────────────────────────────────────────────────────

class FeatureGLBBuilder(GLBBuilder):
    """Extends GLBBuilder with URI-based WebP material support and
    generated vertex tangents for normal-mapped meshes.
    """

    def __init__(self, texture_cache: TextureCache, glb_dir: str):
        """texture_cache: shared cache for this build run.
        glb_dir:       absolute path to the directory where this GLB will
                       live — used to compute the relative URI from GLB to
                       the _textures/ folder.
        """
        super().__init__()
        self.images = []    # glTF images[]
        self.textures = []  # glTF textures[]
        self.samplers = []  # glTF samplers[]
        self.texture_cache = texture_cache
        self.glb_dir = glb_dir
        # Per-builder dedup: same source path within one GLB -> same tex index
        self._tex_idx_by_filename: Dict[str, int] = {}
        # Track which materials use a normal map (so add_piece_mesh knows
        # whether to emit tangents)
        self._materials_need_tangents: Dict[int, bool] = {}
        # Asset-specific flip: rocks30's s3o files were authored with
        # V=0 at the BOTTOM of the atlas (inverse of the default BAR
        # convention where V=0 is at the top). Without this flip, every
        # rocks30 UV shell lands in the wrong tile of its 3×2 atlas.
        # Other features (ad0/trees/etc.) pass V through unchanged.
        self._flip_v = False

    # -- Geometry override: unit-normalize normals + optional tangents ------

    def add_piece_mesh(self, piece, material_idx):
        tri_indices = piece.triangle_indices()
        if len(piece.vertices) == 0 or len(tri_indices) == 0:
            return None

        positions = np.array(
            [[v.x, v.y, v.z] for v in piece.vertices], dtype=np.float32
        )
        normals = np.array(
            [[v.nx, v.ny, v.nz] for v in piece.vertices], dtype=np.float32
        )
        indices_arr = np.array(tri_indices, dtype=np.uint32)

        # Phase 1: fix NaN/zero-length normals via face-normal accumulation
        lengths = np.linalg.norm(normals, axis=1)
        bad_mask = np.isnan(normals).any(axis=1) | (lengths < 1e-8)
        if bad_mask.any():
            face_accum = np.zeros_like(normals)
            for t in range(0, len(indices_arr), 3):
                i0, i1, i2 = int(indices_arr[t]), int(indices_arr[t+1]), int(indices_arr[t+2])
                e1 = positions[i1] - positions[i0]
                e2 = positions[i2] - positions[i0]
                fn = np.cross(e1, e2)
                fl = np.linalg.norm(fn)
                if fl > 1e-12:
                    fn = fn / fl
                for vi in (i0, i1, i2):
                    if bad_mask[vi]:
                        face_accum[vi] += fn
            for vi in np.where(bad_mask)[0]:
                nl = np.linalg.norm(face_accum[vi])
                if nl > 1e-12:
                    normals[vi] = face_accum[vi] / nl
                else:
                    normals[vi] = [0.0, 1.0, 0.0]

        # Phase 2: unit-normalize all normals (glTF spec requires length 1).
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        lengths = np.where(lengths < 1e-8, 1.0, lengths)
        normals = normals / lengths

        # V is normally used verbatim (no 1-t flip). Per the BAR engine
        # source: Recoil loads TGAs with IL_ORIGIN_UPPER_LEFT and uploads
        # bytes unflipped, and most s3o files store UV with V=0 at the
        # image-file top row. glTF's texture convention is also V=0 at
        # top, so the s3o V passes through unchanged for ad0 trees,
        # fir_tree, crystals, etc. — the assets that previously broke
        # when we tried a blanket flip.
        #
        # rocks30 is the exception: its s3o files were authored with V=0
        # at the BOTTOM of the atlas, so without this per-asset flip every
        # UV shell lands in the wrong tile of its 3×2 atlas (empirically
        # verified by overlaying UVs on the atlas image).
        if self._flip_v:
            texcoords = np.array(
                [[v.s, 1.0 - v.t] for v in piece.vertices], dtype=np.float32
            )
        else:
            texcoords = np.array(
                [[v.s, v.t] for v in piece.vertices], dtype=np.float32
            )

        pos_min = positions.min(axis=0).tolist()
        pos_max = positions.max(axis=0).tolist()

        pos_bv = self.add_buffer_view(positions.tobytes(), target=34962)
        pos_acc = self.add_accessor(
            pos_bv, 5126, len(positions), "VEC3",
            min_vals=pos_min, max_vals=pos_max
        )
        norm_bv = self.add_buffer_view(normals.astype(np.float32).tobytes(), target=34962)
        norm_acc = self.add_accessor(norm_bv, 5126, len(normals), "VEC3")
        uv_bv = self.add_buffer_view(texcoords.tobytes(), target=34962)
        uv_acc = self.add_accessor(uv_bv, 5126, len(texcoords), "VEC2")
        idx_bv = self.add_buffer_view(indices_arr.tobytes(), target=34963)
        idx_acc = self.add_accessor(
            idx_bv, 5125, len(indices_arr), "SCALAR",
            min_vals=[int(indices_arr.min())],
            max_vals=[int(indices_arr.max())]
        )

        attributes = {
            "POSITION": pos_acc,
            "NORMAL": norm_acc,
            "TEXCOORD_0": uv_acc,
        }

        # Generate tangents if this material has a normal texture.
        if self._materials_need_tangents.get(material_idx, False):
            tangents = _compute_tangents(positions, normals, texcoords, indices_arr)
            tan_bv = self.add_buffer_view(tangents.tobytes(), target=34962)
            tan_acc = self.add_accessor(tan_bv, 5126, len(tangents), "VEC4")
            attributes["TANGENT"] = tan_acc

        mesh = {
            "name": piece.name,
            "primitives": [{
                "attributes": attributes,
                "indices": idx_acc,
                "material": material_idx,
                "mode": 4,  # TRIANGLES
            }],
        }
        idx = len(self.meshes)
        self.meshes.append(mesh)
        return idx

    # -- Texture / material support ----------------------------------------

    def _add_default_sampler(self) -> int:
        if self.samplers:
            return 0
        self.samplers.append({
            "magFilter": 9729,   # LINEAR
            "minFilter": 9987,   # LINEAR_MIPMAP_LINEAR
            "wrapS": 10497,      # REPEAT
            "wrapT": 10497,      # REPEAT
        })
        return 0

    def _add_uri_texture(self, source_path: str, lossless: bool,
                         match_size_of: Optional[str] = None,
                         keep_full_res: bool = False) -> Optional[int]:
        """Register a source image with the cache and add a glTF texture
        whose image.uri is relative to this GLB's directory.
        Returns the texture index, or None if the source can't be loaded.
        """
        filename = self.texture_cache.register(
            source_path, lossless=lossless,
            match_size_of=match_size_of, keep_full_res=keep_full_res,
        )
        if not filename:
            return None

        # Per-builder dedup: same filename -> same texture index in this GLB
        if filename in self._tex_idx_by_filename:
            return self._tex_idx_by_filename[filename]

        abs_tex_path = os.path.join(self.texture_cache.textures_dir, filename)
        rel = os.path.relpath(abs_tex_path, self.glb_dir).replace('\\', '/')

        img_idx = len(self.images)
        self.images.append({
            "uri": rel,
            "mimeType": "image/webp",
        })

        sampler_idx = self._add_default_sampler()
        tex_idx = len(self.textures)
        self.textures.append({
            "sampler": sampler_idx,
            "source": img_idx,
        })
        self._tex_idx_by_filename[filename] = tex_idx
        return tex_idx

    def add_textured_material(self,
                               color_path: Optional[str],
                               normal_path: Optional[str],
                               extra_path: Optional[str] = None,
                               name: str = "FeatureMat") -> int:
        """Create a PBR material for a feature model.

        color_path : BAR tex1 — diffuse RGB
        normal_path: normal map (RGB = normal, alpha ignored)
        extra_path : BAR tex2 — spec/emit/team RGB, and an alpha channel that
                     is the actual feature alpha-mask. When tex1's own alpha
                     is unusable, we splice tex2's alpha onto tex1 so
                     alphaMode=MASK works correctly for foliage/etc.
        """
        # Per-asset texture policy. rocks30 is kept at full source resolution
        # because its atlas has very small UV shells per model — halving the
        # texture makes each rock look like a featureless blur and hides
        # whether UVs are hitting the right tile at all.
        keep_full_res = 'rocks30' in name.lower()

        mat: dict = {
            "name": name,
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.9,
            },
            "doubleSided": True,
        }

        # Color texture (possibly with alpha borrowed from the Extra texture)
        if color_path:
            filename, has_mask = self.texture_cache.register_color_with_mask(
                color_path, mask_source_path=extra_path,
                keep_full_res=keep_full_res,
            )
            if filename:
                color_tex = self._register_cache_filename(filename)
                mat["pbrMetallicRoughness"]["baseColorTexture"] = {"index": color_tex}
                if has_mask:
                    mat["alphaMode"] = "MASK"
                    mat["alphaCutoff"] = 0.5

        # Normal texture — clamped to the color texture's dimensions when
        # the source normal is larger, so both textures sample at the same
        # per-UV-texel rate. Without this, rocks30 (3072×2048 normal over a
        # 1536×1024 color) produced "blobby" shading that didn't correspond
        # to the diffuse detail.
        has_normal = False
        if normal_path:
            normal_tex = self._add_uri_texture(
                normal_path, lossless=True, match_size_of=color_path,
                keep_full_res=keep_full_res,
            )
            if normal_tex is not None:
                mat["normalTexture"] = {"index": normal_tex}
                has_normal = True

        idx = len(self.materials)
        self.materials.append(mat)
        self._materials_need_tangents[idx] = has_normal
        return idx

    def _register_cache_filename(self, filename: str) -> int:
        """Given a filename already living in the texture cache directory,
        add an image+texture entry (or return existing) and return the
        texture index. Used by register_color_with_mask, which does its own
        cache interaction before handing back a filename.
        """
        if filename in self._tex_idx_by_filename:
            return self._tex_idx_by_filename[filename]

        abs_tex_path = os.path.join(self.texture_cache.textures_dir, filename)
        rel = os.path.relpath(abs_tex_path, self.glb_dir).replace('\\', '/')

        img_idx = len(self.images)
        self.images.append({
            "uri": rel,
            "mimeType": "image/webp",
        })
        sampler_idx = self._add_default_sampler()
        tex_idx = len(self.textures)
        self.textures.append({
            "sampler": sampler_idx,
            "source": img_idx,
        })
        self._tex_idx_by_filename[filename] = tex_idx
        return tex_idx

    # -- Final GLB assembly -------------------------------------------------

    def build_glb(self) -> bytes:
        import struct
        import json

        gltf = {
            "asset": {
                "version": "2.0",
                "generator": "BAR-Map-Sync-Features",
            },
            "scene": 0,
            "scenes": self.scenes,
            "nodes": self.nodes,
            "meshes": self.meshes,
            "accessors": self.accessors,
            "bufferViews": self.buffer_views,
            "buffers": [{"byteLength": len(self.buffer_data)}],
        }
        if self.materials:
            gltf["materials"] = self.materials
        if self.images:
            gltf["images"] = self.images
        if self.textures:
            gltf["textures"] = self.textures
        if self.samplers:
            gltf["samplers"] = self.samplers

        json_str = json.dumps(gltf, separators=(',', ':'))
        json_bytes = json_str.encode('utf-8')

        json_pad = (4 - len(json_bytes) % 4) % 4
        json_bytes += b' ' * json_pad

        bin_data = bytes(self.buffer_data)
        bin_pad = (4 - len(bin_data) % 4) % 4
        bin_data += b'\x00' * bin_pad

        total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_data)

        glb = bytearray()
        glb += struct.pack('<I', 0x46546C67)  # "glTF"
        glb += struct.pack('<I', 2)
        glb += struct.pack('<I', total_length)
        glb += struct.pack('<I', len(json_bytes))
        glb += struct.pack('<I', 0x4E4F534A)  # "JSON"
        glb += json_bytes
        glb += struct.pack('<I', len(bin_data))
        glb += struct.pack('<I', 0x004E4942)  # "BIN\0"
        glb += bin_data
        return bytes(glb)


# ──────────────────────────────────────────────────────────────────────────
# Tangent computation (MikkTSpace-like, good enough for feature models)
# ──────────────────────────────────────────────────────────────────────────

def _compute_tangents(positions: np.ndarray,
                       normals: np.ndarray,
                       texcoords: np.ndarray,
                       indices: np.ndarray) -> np.ndarray:
    """Compute per-vertex tangents as VEC4 (xyz=tangent, w=bitangent sign).

    This is the standard Lengyel method: accumulate per-triangle tangents
    weighted into per-vertex accumulators, then Gram-Schmidt orthogonalize
    against the vertex normal. Not bit-exact MikkTSpace, but eliminates the
    MESH_PRIMITIVE_GENERATED_TANGENT_SPACE validator warning and produces
    correct normal-mapped shading for our feature models.
    """
    n_verts = len(positions)
    tan1 = np.zeros((n_verts, 3), dtype=np.float64)
    tan2 = np.zeros((n_verts, 3), dtype=np.float64)

    tris = indices.reshape(-1, 3)
    for i0, i1, i2 in tris:
        i0, i1, i2 = int(i0), int(i1), int(i2)
        v0, v1, v2 = positions[i0], positions[i1], positions[i2]
        w0, w1, w2 = texcoords[i0], texcoords[i1], texcoords[i2]

        x1 = v1[0] - v0[0]
        x2 = v2[0] - v0[0]
        y1 = v1[1] - v0[1]
        y2 = v2[1] - v0[1]
        z1 = v1[2] - v0[2]
        z2 = v2[2] - v0[2]

        s1 = w1[0] - w0[0]
        s2 = w2[0] - w0[0]
        t1 = w1[1] - w0[1]
        t2 = w2[1] - w0[1]

        denom = s1 * t2 - s2 * t1
        if abs(denom) < 1e-20:
            continue
        r = 1.0 / denom

        sdir = np.array([
            (t2 * x1 - t1 * x2) * r,
            (t2 * y1 - t1 * y2) * r,
            (t2 * z1 - t1 * z2) * r,
        ], dtype=np.float64)
        tdir = np.array([
            (s1 * x2 - s2 * x1) * r,
            (s1 * y2 - s2 * y1) * r,
            (s1 * z2 - s2 * z1) * r,
        ], dtype=np.float64)

        tan1[i0] += sdir; tan1[i1] += sdir; tan1[i2] += sdir
        tan2[i0] += tdir; tan2[i1] += tdir; tan2[i2] += tdir

    tangents = np.zeros((n_verts, 4), dtype=np.float32)
    for i in range(n_verts):
        n = normals[i].astype(np.float64)
        t = tan1[i]
        # Gram-Schmidt orthogonalize
        t_ortho = t - n * np.dot(n, t)
        tl = np.linalg.norm(t_ortho)
        if tl < 1e-12:
            # Degenerate — pick an arbitrary tangent orthogonal to n
            ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            t_ortho = np.cross(n, ref)
            tl = np.linalg.norm(t_ortho)
            if tl < 1e-12:
                tangents[i] = [1.0, 0.0, 0.0, 1.0]
                continue
        t_ortho = t_ortho / tl
        # Handedness
        cross_nt = np.cross(n, t)
        w = -1.0 if np.dot(cross_nt, tan2[i]) < 0.0 else 1.0
        tangents[i, 0] = t_ortho[0]
        tangents[i, 1] = t_ortho[1]
        tangents[i, 2] = t_ortho[2]
        tangents[i, 3] = w
    return tangents


# ──────────────────────────────────────────────────────────────────────────
# Public build function
# ──────────────────────────────────────────────────────────────────────────

def build_feature_glb(s3o_path: str,
                       color_tex_path: Optional[str],
                       normal_tex_path: Optional[str],
                       texture_cache: TextureCache,
                       glb_out_dir: str,
                       extra_tex_path: Optional[str] = None) -> Optional[bytes]:
    """Parse an .s3o and build a textured .glb referencing the shared
    TextureCache. Returns GLB bytes, or None on failure.

    extra_tex_path is BAR's tex2/"Extra" texture. Its RGB (spec/emit) is
    ignored, but its *alpha channel* carries the feature mask for
    transparency (e.g. leaves on trees). When provided and the color
    texture has no usable alpha of its own, that alpha gets spliced onto
    the color texture so alphaMode=MASK works correctly.
    """
    try:
        model = parse_s3o(s3o_path)
    except Exception as e:
        print(f"      [S3O] Parse error {os.path.basename(s3o_path)}: {e}")
        return None

    if model.root_piece is None:
        print(f"      [S3O] {os.path.basename(s3o_path)}: no root piece")
        return None

    builder = FeatureGLBBuilder(texture_cache=texture_cache, glb_dir=glb_out_dir)
    asset_name = os.path.splitext(os.path.basename(s3o_path))[0]
    # rocks30 is BAR's only `cuspbr = "yes"` feature and is authored with
    # V=0 at the atlas bottom (inverse of the usual s3o convention). Flip
    # V here so each model's UVs land in the correct 3×2 atlas tile.
    if 'rocks30' in asset_name.lower():
        builder._flip_v = True
    mat_idx = builder.add_textured_material(
        color_path=color_tex_path,
        normal_path=normal_tex_path,
        extra_path=extra_tex_path,
        name=asset_name,
    )
    root_node = builder.add_piece_node(model.root_piece, mat_idx)
    builder.scenes[0]["nodes"] = [root_node]

    return builder.build_glb()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python glb_feature_builder.py <file.s3o> [color] [normal] [out.glb]")
        sys.exit(1)
    s3o = sys.argv[1]
    color = sys.argv[2] if len(sys.argv) > 2 else None
    normal = sys.argv[3] if len(sys.argv) > 3 else None
    out = sys.argv[4] if len(sys.argv) > 4 else (os.path.splitext(s3o)[0] + '.glb')

    out_dir = os.path.dirname(os.path.abspath(out))
    textures_dir = os.path.join(out_dir, '_textures')
    cache = TextureCache(textures_dir)
    data = build_feature_glb(s3o, color, normal, cache, out_dir)
    if data is None:
        print("Conversion failed")
        sys.exit(2)
    with open(out, 'wb') as f:
        f.write(data)
    print(f"Written: {out} ({len(data):,} bytes)")
    print(f"Textures: {cache.stats}")
