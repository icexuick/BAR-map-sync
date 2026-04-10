"""
GLB Feature Builder — converts a parsed S3OModel + texture files into a
geometry-only .glb plus per-map WebP textures stored separately.

Output layout:
    features/
      fir_tree_tall_5__tree_fir_tall_1/
        model.glb                 # geometry-only, no images/textures
                                  # material.extras has texture filenames
    maps_features/
      ad0_fir/                    # grouped by feature texture set
        ad0_fir_1_masked__angel-crossing.webp
        ad0_fir_1_masked__boreal-falls.webp
        ad0_fir_normal__angel-crossing.webp
        ad0_fir_normal__boreal-falls.webp
      rocks30/
        rocks30_1__folsom-dam.webp
        rocks30_normal__folsom-dam.webp

Key design points:
  - GLBs are geometry-only (shared across all maps). Materials carry
    texture filenames in glTF extras (colorTex, normalTex) so the viewer
    can load per-map textures at runtime.
  - TextureCache writes per-map textures into maps_features/<feature_group>/
    with an __<mapslug> suffix on every file. This makes it easy to compare
    texture variants across maps and optionally merge them by hand.
  - Within a single map run, source-path and content-hash dedup still
    apply (e.g. 55 fir variants sharing one atlas → one webp written).
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
COLOR_WEBP_QUALITY = 70


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
# TextureCache: per-map textures grouped by feature, __mapslug suffix
# ──────────────────────────────────────────────────────────────────────────

class TextureCache:
    """Writes per-map WebP textures into maps_features/<feature_group>/
    with an __<mapslug> suffix on every filename.

    Usage:
        cache = TextureCache(
            textures_root="maps_features",
            map_slug="angel-crossing",
        )
        filename = cache.register(
            source_path="...tex1.dds", lossless=False,
            feature_group="ad0_fir",
        )
        # => "ad0_fir_1__angel-crossing.webp"
        # written to maps_features/ad0_fir/ad0_fir_1__angel-crossing.webp

    Within a single run, source-path dedup and content-hash dedup apply
    (e.g. 55 fir variants sharing one atlas → one webp written).
    No cross-map dedup — each map always gets its own copy.

    Not thread-safe. One instance per run.
    """

    def __init__(self, textures_root: str, map_slug: str):
        self.textures_root = textures_root
        self.map_slug = map_slug

        # Maps: source_key -> (filename, feature_group)
        self._by_source: Dict[str, Tuple[str, str]] = {}
        # Maps: content_sha1 -> (filename, feature_group)
        self._by_hash: Dict[str, Tuple[str, str]] = {}

        self.stats = {
            'unique_files': 0,
            'reused_by_source': 0,
            'reused_by_content': 0,
            'total_bytes': 0,
        }

    def register(self, source_path: str, lossless: bool,
                 feature_group: str,
                 match_size_of: Optional[str] = None,
                 keep_full_res: bool = False,
                 shared: bool = False) -> Optional[str]:
        """Encode + store the texture, returning the filename (no dir prefix).
        Returns None if the source can't be loaded.

        feature_group: the feature texture set name (e.g. "ad0_fir",
            "rocks30"). Used as the subdirectory under textures_root.
        shared: if True, write without __<mapslug> suffix (e.g. normal maps
            that are identical across all maps).
        """
        if not source_path:
            return None
        key = os.path.normcase(os.path.abspath(source_path))
        if match_size_of:
            key = key + '|match=' + os.path.normcase(os.path.abspath(match_size_of))
        if keep_full_res:
            key = key + '|full'
        cached = self._by_source.get(key)
        if cached:
            self.stats['reused_by_source'] += 1
            return cached[0]

        img = _load_image_any(source_path)
        if img is None:
            return None

        if match_size_of:
            partner = _load_image_any(match_size_of)
            if partner is not None and partner.size != img.size:
                tw, th = partner.size
                if tw * th < img.width * img.height:
                    img = img.resize((tw, th), Image.Resampling.LANCZOS)

        return self._store_image(img, source_path, lossless, key,
                                  feature_group=feature_group,
                                  keep_full_res=keep_full_res,
                                  shared=shared)

    def register_color_with_mask(self,
                                  color_path: str,
                                  mask_source_path: Optional[str],
                                  feature_group: str,
                                  keep_full_res: bool = False,
                                  shared: bool = False) -> Tuple[Optional[str], bool]:
        """Register a color texture, optionally borrowing its alpha channel
        from another image (BAR's tex2/"Extra") when the color's own alpha
        is unusable.

        Returns (filename, has_mask):
          filename: the webp filename (no directory prefix)
          has_mask: True if an alpha mask is now present
        """
        if not color_path:
            return None, False

        color_img = _load_image_any(color_path)
        if color_img is None:
            return None, False

        # Check if the alpha channel is a usable mask (has pixels > 128).
        # Some textures (unit wrecks) have an alpha channel that stores
        # specular/team-color data with all values < 128, which would make
        # the entire mesh invisible with alphaTest=0.5.
        color_has_alpha = False
        if 'A' in color_img.getbands():
            import numpy as np
            alpha = np.array(color_img.split()[-1])
            if np.any(alpha > 128):
                color_has_alpha = True

        merged = False
        if not color_has_alpha and mask_source_path:
            raw = _load_raw_rgba(mask_source_path)
            if raw is not None:
                mask = _derive_mask_from_extra(raw)
                if mask is not None:
                    if mask.size != color_img.size:
                        mask = mask.resize(color_img.size, Image.Resampling.LANCZOS)
                    rgb = color_img.convert('RGB')
                    r, g, b = rgb.split()
                    color_img = Image.merge('RGBA', (r, g, b, mask))
                    merged = True

        has_mask = color_has_alpha or merged

        key_parts = [os.path.normcase(os.path.abspath(color_path))]
        if merged:
            key_parts.append(os.path.normcase(os.path.abspath(mask_source_path)))
        key = '|masked|'.join(key_parts)
        if keep_full_res:
            key = key + '|full'

        cached = self._by_source.get(key)
        if cached:
            self.stats['reused_by_source'] += 1
            return cached[0], has_mask

        filename = self._store_image(
            color_img, color_path, lossless=False, source_key=key,
            feature_group=feature_group,
            name_suffix='_masked' if merged else '',
            keep_full_res=keep_full_res,
            shared=shared,
        )
        return filename, has_mask

    def _store_image(self,
                      img: Image.Image,
                      source_path_for_name: str,
                      lossless: bool,
                      source_key: str,
                      feature_group: str,
                      name_suffix: str = '',
                      keep_full_res: bool = False,
                      shared: bool = False) -> Optional[str]:
        """Encode to webp, write to feature_group dir.

        shared=False (default): per-map texture with __<mapslug> suffix.
        shared=True: map-independent texture (e.g. normal maps) without
            slug suffix. Skips writing if the file already exists on disk.
        """
        data = _encode_webp(img, lossless=lossless, keep_full_res=keep_full_res)
        content_hash = hashlib.sha1(data).hexdigest()

        # Content-hash dedup within this run
        existing = self._by_hash.get(content_hash)
        if existing:
            self._by_source[source_key] = existing
            self.stats['reused_by_content'] += 1
            return existing[0]

        stem = _safe_basename(source_path_for_name) + name_suffix
        if shared:
            filename = f"{stem}.webp"
        else:
            filename = f"{stem}__{self.map_slug}.webp"

        out_dir = os.path.join(self.textures_root, feature_group)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filename)

        # Shared textures: skip writing if already on disk (from earlier run)
        if shared and os.path.isfile(out_path):
            entry = (filename, feature_group)
            self._by_source[source_key] = entry
            self._by_hash[content_hash] = entry
            self.stats['reused_by_content'] += 1
            return filename

        with open(out_path, 'wb') as f:
            f.write(data)

        entry = (filename, feature_group)
        self._by_source[source_key] = entry
        self._by_hash[content_hash] = entry
        self.stats['unique_files'] += 1
        self.stats['total_bytes'] += len(data)
        return filename


# ──────────────────────────────────────────────────────────────────────────
# FeatureGLBBuilder: extends parent with textured material + tangent support
# ──────────────────────────────────────────────────────────────────────────

class FeatureGLBBuilder(GLBBuilder):
    """Extends GLBBuilder with geometry-only GLB output.

    Materials carry texture filenames in glTF extras (colorTex, normalTex)
    so the viewer can load per-map textures at runtime. No images/textures/
    samplers are emitted in the glTF JSON.
    """

    def __init__(self):
        super().__init__()
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

    # -- Material support (geometry-only — texture filenames in extras) ------

    def add_textured_material(self,
                               color_tex_filename: Optional[str] = None,
                               normal_tex_filename: Optional[str] = None,
                               has_mask: bool = False,
                               feature_group: Optional[str] = None,
                               name: str = "FeatureMat") -> int:
        """Create a PBR material for a feature model (geometry-only GLB).

        Instead of glTF texture references, texture filenames are stored in
        material.extras so the viewer can load per-map textures at runtime.
        """
        mat: dict = {
            "name": name,
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.9,
            },
            "doubleSided": True,
        }

        if has_mask:
            mat["alphaMode"] = "MASK"
            mat["alphaCutoff"] = 0.5

        # Store texture filenames in extras for the viewer to resolve
        extras: dict = {}
        if color_tex_filename:
            extras["colorTex"] = color_tex_filename
        if normal_tex_filename:
            extras["normalTex"] = normal_tex_filename
        if feature_group:
            extras["featureGroup"] = feature_group
        # Signal to the viewer that UVs were V-flipped at build time
        # (rocks30 / cuspbr assets). The viewer must set flipY=false
        # on textures for these materials to avoid a double-flip.
        if self._flip_v:
            extras["uvFlipV"] = True
        if extras:
            mat["extras"] = extras

        has_normal = normal_tex_filename is not None
        idx = len(self.materials)
        self.materials.append(mat)
        self._materials_need_tangents[idx] = has_normal
        return idx

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
        # Geometry-only GLB: no images/textures/samplers.
        # Texture filenames live in material.extras for the viewer.

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

def register_feature_textures(
    color_tex_path: Optional[str],
    normal_tex_path: Optional[str],
    extra_tex_path: Optional[str],
    texture_cache: TextureCache,
    feature_group: str,
    asset_name: str,
) -> dict:
    """Register textures with the per-map cache. Returns a dict with
    keys 'color_filename', 'normal_filename', 'has_mask', 'feature_group'
    for use by the GLB builder or when the GLB already exists.
    """
    is_rocks30 = 'rocks30' in asset_name.lower()
    keep_full_res = is_rocks30
    # rocks30 textures come from BAR.sdd and are map-independent
    shared_color = is_rocks30

    color_filename = None
    has_mask = False
    if color_tex_path:
        color_filename, has_mask = texture_cache.register_color_with_mask(
            color_tex_path, mask_source_path=extra_tex_path,
            feature_group=feature_group,
            keep_full_res=keep_full_res,
            shared=shared_color,
        )

    normal_filename = None
    if normal_tex_path:
        normal_filename = texture_cache.register(
            normal_tex_path, lossless=True,
            feature_group=feature_group,
            match_size_of=color_tex_path,
            keep_full_res=keep_full_res,
            shared=True,  # normals are map-independent, no __mapslug suffix
        )

    return {
        'color_filename': color_filename,
        'normal_filename': normal_filename,
        'has_mask': has_mask,
        'feature_group': feature_group,
    }


def build_feature_glb(s3o_path: str,
                       color_tex_filename: Optional[str] = None,
                       normal_tex_filename: Optional[str] = None,
                       has_mask: bool = False,
                       feature_group: Optional[str] = None) -> Optional[bytes]:
    """Parse an .s3o and build a geometry-only .glb. Texture filenames
    (already registered with TextureCache) are stored in material.extras.
    Returns GLB bytes, or None on failure.
    """
    try:
        model = parse_s3o(s3o_path)
    except Exception as e:
        print(f"      [S3O] Parse error {os.path.basename(s3o_path)}: {e}")
        return None

    if model.root_piece is None:
        print(f"      [S3O] {os.path.basename(s3o_path)}: no root piece")
        return None

    builder = FeatureGLBBuilder()
    asset_name = os.path.splitext(os.path.basename(s3o_path))[0]
    # rocks30 is BAR's only `cuspbr = "yes"` feature and is authored with
    # V=0 at the atlas bottom (inverse of the usual s3o convention). Flip
    # V here so each model's UVs land in the correct 3×2 atlas tile.
    if 'rocks30' in asset_name.lower():
        builder._flip_v = True
    mat_idx = builder.add_textured_material(
        color_tex_filename=color_tex_filename,
        normal_tex_filename=normal_tex_filename,
        has_mask=has_mask,
        feature_group=feature_group,
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
    cache = TextureCache(textures_root=textures_dir, map_slug='standalone')
    feature_group = os.path.splitext(os.path.basename(s3o))[0]
    tex_info = register_feature_textures(
        color, normal, None, cache, feature_group, feature_group)
    data = build_feature_glb(
        s3o,
        color_tex_filename=tex_info['color_filename'],
        normal_tex_filename=tex_info['normal_filename'],
        has_mask=tex_info['has_mask'],
        feature_group=tex_info['feature_group'],
    )
    if data is None:
        print("Conversion failed")
        sys.exit(2)
    with open(out, 'wb') as f:
        f.write(data)
    print(f"Written: {out} ({len(data):,} bytes)")
    print(f"Textures: {cache.stats}")
