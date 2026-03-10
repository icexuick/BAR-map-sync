import requests
import py7zr
import re
import os
import shutil
import time
import ftplib
import struct
import io
import sys
import argparse
from dotenv import load_dotenv

import numpy as np
import imageio.v3 as iio
from PIL import Image

# --- CRASH PROTECTION ---
Image.MAX_IMAGE_PIXELS = None 

# --- CONFIGURATION ---
load_dotenv()

WEBFLOW_API_TOKEN = os.environ.get("WEBFLOW_API_TOKEN")
FTP_HOST = os.environ.get("FTP_HOST")
FTP_USER = os.environ.get("FTP_USER")
FTP_PASSWORD = os.environ.get("FTP_PASSWORD")
FTP_PATH = os.environ.get("FTP_PATH")         
FTP_BASE_URL = os.environ.get("FTP_BASE_URL") 
GITHUB_LAVA_URL = "https://api.github.com/repos/beyond-all-reason/Beyond-All-Reason/contents/common/configs/LavaMaps"

if not WEBFLOW_API_TOKEN:
    raise ValueError("CRITICAL: No WEBFLOW_API_TOKEN found.")

COLLECTION_ID = "6564c6553676389f8ba45aaf"

# Field Slugs (WEBFLOW)
FIELD_MIN = "map-height-min"
FIELD_MAX = "map-height-max"
FIELD_DOWNLOAD_URL = "downloadurl"
FIELD_VOID_WATER = "void-water" 
FIELD_NORMAL_MAP = "normal-map"
FIELD_SKYBOX = "skybox"
FIELD_LAVA_LEVEL = "lavalevel"
FIELD_TEXTURE_MAP = "mini-map" 
FIELD_HEIGHT_MAP = "height-map"
FIELD_METAL_MAP = "metal-map"

# Water & Meta Properties
FIELD_WATER_TINT = "water-lava-color-tint"
FIELD_WATER_BASE = "water-basecolor"
FIELD_WATER_MIN  = "water-min"
FIELD_WATER_ABSORB = "water-absorb"
FIELD_VERSION = "version"

SKYBOX_MAX_WIDTH = 4096  
DIFFUSE_MAX_WIDTH = 4096 
MAX_PIXEL_DIMENSION = 4096

HEADERS_WEBFLOW = {
    "Authorization": f"Bearer {WEBFLOW_API_TOKEN}",
    "accept-version": "2.0.0",
    "content-type": "application/json"
}

MAX_FILE_SIZE_MB = 4

# --- LOGIC SETTINGS ---
FORCE_VERSION_OVERWRITE = True
FORCE_CORE_OVERWRITE = True 
FORCE_HEAVY_OVERWRITE = False
FORCE_METADATA_UPDATE = True

# --- LUA COMMENT STRIPPER ---
def strip_lua_comments(content):
    """Verwijder alle single-line Lua comments (-- ...) uit de content."""
    lines = content.split('\n')
    cleaned = []
    for line in lines:
        stripped = re.sub(r'--.*$', '', line)
        cleaned.append(stripped)
    return '\n'.join(cleaned)

# --- TEXT/COLOR HELPERS ---
def rgb_float_to_hex(rgb_string):
    try:
        matches = re.findall(r'([\d\.]+)', rgb_string)
        if len(matches) < 3: return None
        r = int(float(matches[0]) * 255)
        g = int(float(matches[1]) * 255)
        b = int(float(matches[2]) * 255)
        r = max(0, min(255, r)); g = max(0, min(255, g)); b = max(0, min(255, b))
        return f"#{r:02x}{g:02x}{b:02x}"
    except: return None

def extract_lua_value(content, block_name, keys_to_find):
    lines = content.split('\n')
    in_block = False
    
    for line in lines:
        clean_line = line.strip()
        if clean_line.startswith('--'): continue
        if re.search(rf'^\s*{block_name}\s*=\s*\{{', clean_line, re.IGNORECASE):
            in_block = True
            continue
        if in_block:
            code_part = clean_line.split('--')[0] 
            for key in keys_to_find:
                if key.lower() in code_part.lower():
                    val_match = re.search(r'\{([^}]+)\}', code_part)
                    if val_match: return val_match.group(1).strip()
    return None

def extract_version_from_lua(content):
    lines = content.split('\n')
    for line in lines:
        clean_line = line.strip()
        if clean_line.startswith('--'): continue
        if 'version' in clean_line.lower():
            match = re.search(r'version\s*=\s*[\'"]([^\'"]+)[\'"]', clean_line, re.IGNORECASE)
            if match:
                return match.group(1)
    return None

# --- DXT1 DECOMPRESSOR ---
def dxt1_decompress_tile(data, width, height):
    out_rgba = bytearray(width * height * 4)
    block_count_x = width // 4
    block_count_y = height // 4
    
    expected_bytes = block_count_x * block_count_y * 8
    if len(data) < expected_bytes: return out_rgba

    for by in range(block_count_y):
        for bx in range(block_count_x):
            block_idx = by * block_count_x + bx
            offset = block_idx * 8
            block = data[offset : offset+8]
            c0 = block[0] | (block[1] << 8); c1 = block[2] | (block[3] << 8)
            
            def unpack_565(c):
                r = ((c & 0xF800) >> 8); g = ((c & 0x07E0) >> 3); b = ((c & 0x001F) << 3)
                return (r, g, b)
            
            r0, g0, b0 = unpack_565(c0); r1, g1, b1 = unpack_565(c1)
            colors = [None]*4
            colors[0] = (r0, g0, b0, 255); colors[1] = (r1, g1, b1, 255)
            
            if c0 > c1:
                colors[2] = ((2*r0+r1)//3, (2*g0+g1)//3, (2*b0+b1)//3, 255)
                colors[3] = ((r0+2*r1)//3, (g0+2*g1)//3, (b0+2*b1)//3, 255)
            else:
                colors[2] = ((r0+r1)//2, (g0+g1)//2, (b0+b1)//2, 255)
                colors[3] = (0, 0, 0, 0)
            
            for y in range(4):
                row_byte = block[4 + y]
                for x in range(4):
                    code = (row_byte >> (x * 2)) & 3
                    pixel = colors[code]
                    idx = ((by * 4 + y) * width + (bx * 4 + x)) * 4
                    out_rgba[idx:idx+4] = pixel
    return out_rgba

# --- SPRING MAP PARSER ---
class SpringMapParser:
    @staticmethod
    def get_filenames_from_lua(lua_content):
        smf_match = re.search(r'smf\s*=\s*\{[^}]*smtFileName0\s*=\s*["\']([^"\']+)["\']', lua_content, re.IGNORECASE | re.DOTALL)
        if not smf_match:
             smf_match = re.search(r'smtFileName0\s*=\s*["\']([^"\']+)["\']', lua_content, re.IGNORECASE)
        return smf_match.group(1) if smf_match else None

    @staticmethod
    def parse_smf_header(f):
        f.seek(0)
        header_data = f.read(128)
        if len(header_data) < 80: return None
        magic = header_data[:16].rstrip(b'\x00')
        if magic != b'spring map file': return None
        
        ints = struct.unpack('<20I', header_data[:80])
        width_val = ints[6]; height_val = ints[7] 
        width_units = width_val // 128
        height_units = height_val // 128
            
        if width_units == 0 or height_units == 0: return None

        ptr_start = 13
        header = {
            "mapWidth": width_units * 128, 
            "widthUnits": width_units,
            "heightUnits": height_units,
            "heightMapIndex": ints[ptr_start],
            "typeMapIndex": ints[ptr_start+1],
            "tileIndexMapIndex": ints[ptr_start+2],
            "miniMapIndex": ints[ptr_start+3],
            "metalMapIndex": ints[ptr_start+4]
        }
        print(f"      [MapParser] Header Info: {header['mapWidth']}x{height_units*128} px")
        return header

    @staticmethod
    def extract_heightmap(f, header):
        print("      [MapParser] Extracting Heightmap...")
        width = header["widthUnits"] * 128 + 1
        height = header["heightUnits"] * 128 + 1
        size_bytes = width * height * 2 
        if header["heightMapIndex"] == 0: return None
        f.seek(header["heightMapIndex"])
        buf = f.read(size_bytes)
        if len(buf) < size_bytes: return None
        arr = np.frombuffer(buf, dtype=np.uint16).reshape((height, width))
        arr_8bit = (arr >> 8).astype(np.uint8) 
        return Image.fromarray(arr_8bit, mode='L')

    @staticmethod
    def extract_metalmap(f, header):
        print("      [MapParser] Extracting Metalmap...")
        width = header["widthUnits"] * 64
        height = header["heightUnits"] * 64
        size_bytes = width * height 
        offset = header["metalMapIndex"]
        if offset == 0: offset = header["typeMapIndex"]
        if offset == 0: return None
        f.seek(offset)
        buf = f.read(size_bytes)
        if len(buf) < size_bytes: return None
        arr = np.frombuffer(buf, dtype=np.uint8).reshape((height, width))
        return Image.fromarray(arr, mode='L')

    @staticmethod
    def read_until_null(f):
        chars = []
        while True:
            c = f.read(1)
            if c == b'\x00' or not c: break
            chars.append(c)
        return b"".join(chars).decode('utf-8', 'ignore')

    @staticmethod
    def extract_diffuse_texture(f_smf, f_smt, header, target_width=4096):
        print("      [MapParser] Stitching Diffuse Texture...")
        cols = header["mapWidth"] // 4
        rows = (header["heightUnits"] * 128) // 4
        num_indices = cols * rows
        
        f_smf.seek(header["tileIndexMapIndex"])
        f_smf.read(12) 
        SpringMapParser.read_until_null(f_smf) 
        
        indices_buf = f_smf.read(num_indices * 4) 
        if len(indices_buf) < num_indices * 4: return None
        tile_indices = np.frombuffer(indices_buf, dtype=np.uint32)
        
        f_smt.seek(0)
        smt_head_buf = f_smt.read(32)
        if len(smt_head_buf) < 32: return None
        
        smt_head = struct.unpack('<16s 4I', smt_head_buf)
        num_smt_tiles = smt_head[2]
        
        f_smt.seek(0, 2)
        file_size = f_smt.tell()
        data_size = file_size - 32
        
        calc_stride = data_size // num_smt_tiles if num_smt_tiles > 0 else 680
        
        if calc_stride >= 512:
            TILE_STRIDE = 680; real_w, real_h = 32, 32; bytes_to_read = 512
        else:
            TILE_STRIDE = calc_stride; bytes_to_read = calc_stride
            if calc_stride >= 128: real_w, real_h = 16, 16
            elif calc_stride >= 32: real_w, real_h = 8, 8
            else: real_w, real_h = 4, 4

        smt_data_start = 32
        decoded_tiles = {}
        unique_indices = np.unique(tile_indices)
        
        success_count = 0
        for tile_id in unique_indices:
            if tile_id >= num_smt_tiles: continue
            offset = smt_data_start + (tile_id * TILE_STRIDE)
            f_smt.seek(offset)
            dxt_data = f_smt.read(bytes_to_read)
            if len(dxt_data) < bytes_to_read: continue

            try:
                rgba_bytes = dxt1_decompress_tile(dxt_data, real_w, real_h)
                img = Image.frombytes("RGBA", (real_w, real_h), bytes(rgba_bytes))
                if real_w < 32: img = img.resize((32, 32), Image.Resampling.NEAREST)
                decoded_tiles[tile_id] = img
                success_count += 1
            except: pass

        if success_count == 0: return None

        full_w = cols * 32
        full_h = rows * 32
        canvas = Image.new("RGBA", (full_w, full_h))
        
        current_idx = 0
        for y in range(rows):
            for x in range(cols):
                t_id = tile_indices[current_idx]
                current_idx += 1
                if t_id in decoded_tiles:
                    canvas.paste(decoded_tiles[t_id], (x * 32, y * 32))
        
        canvas = canvas.convert("RGB")
        if full_w > target_width:
             print(f"      [MapParser] Resizing texture from {full_w}x{full_h} to {target_width}...")
             ratio = target_width / float(full_w)
             new_h = int(full_h * ratio)
             canvas = canvas.resize((target_width, new_h), Image.Resampling.LANCZOS)
        return canvas

    @staticmethod
    def find_file_recursive(root_dir, extension):
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(extension):
                    return os.path.join(root, file)
        return None

    @staticmethod
    def process_map_files(temp_dir, lua_filename, task_flags, smt_filename_hint=None):
        smf_path = SpringMapParser.find_file_recursive(temp_dir, ".smf")
        smt_path = None
        if task_flags.get("diffuse"):
            smt_path = SpringMapParser.find_file_recursive(temp_dir, ".smt")
            
        if not smf_path: 
            print("      [MapParser] Error: SMF file missing.")
            return {}
            
        results = {}
        try:
            with open(smf_path, 'rb') as f_smf:
                header = SpringMapParser.parse_smf_header(f_smf)
                if not header: return {}
                
                if task_flags.get("height"):
                    try:
                        h_img = SpringMapParser.extract_heightmap(f_smf, header)
                        if h_img: results["heightmap"] = h_img
                    except Exception as e: print(f"      [MapParser] Heightmap Error: {e}")

                if task_flags.get("metal"):
                    try:
                        m_img = SpringMapParser.extract_metalmap(f_smf, header)
                        if m_img: results["metalmap"] = m_img
                    except Exception as e: print(f"      [MapParser] Metalmap Error: {e}")
                    
                if task_flags.get("diffuse") and smt_path:
                    try:
                        with open(smt_path, 'rb') as f_smt:
                            d_img = SpringMapParser.extract_diffuse_texture(f_smf, f_smt, header, target_width=DIFFUSE_MAX_WIDTH)
                            if d_img: results["diffuse"] = d_img
                    except Exception as e: 
                        print(f"      [MapParser] Diffuse Error: {e}")

        except Exception as e: print(f"      [MapParser] Critical Error: {e}")
        return results

# --- SKYBOX PROCESSOR ---
class SkyboxProcessor:
    @staticmethod
    def sample_equirectangular(faces, out_width, out_height):
        print(f"      [Skybox] Calculating projection ({out_width}x{out_height})...")
        corrected_faces = []
        for i, face in enumerate(faces):
            if i in [0, 1, 4, 5]: corrected_faces.append(np.flipud(face))
            else: corrected_faces.append(face)
        faces = corrected_faces
        u = np.linspace(0, 1, out_width)
        v = np.linspace(0, 1, out_height)
        uu, vv = np.meshgrid(u, v)
        theta = uu * 2 * np.pi
        phi = vv * np.pi
        x = -np.sin(phi) * np.sin(theta)
        y = np.cos(phi)
        z = -np.sin(phi) * np.cos(theta) 
        absX, absY, absZ = np.abs(x), np.abs(y), np.abs(z)
        isXPositive = x > 0; isYPositive = y > 0; isZPositive = z > 0
        maxAxis = np.maximum(np.maximum(absX, absY), absZ)
        u_face = np.zeros_like(maxAxis); v_face = np.zeros_like(maxAxis); face_idx = np.zeros_like(maxAxis, dtype=int)
        mask = (isXPositive) & (absX >= absY) & (absX >= absZ); face_idx[mask] = 0; u_face[mask] = -z[mask]/absX[mask]; v_face[mask] = y[mask]/absX[mask]
        mask = (~isXPositive) & (absX >= absY) & (absX >= absZ); face_idx[mask] = 1; u_face[mask] = z[mask]/absX[mask]; v_face[mask] = y[mask]/absX[mask]
        mask = (isYPositive) & (absY >= absX) & (absY >= absZ); face_idx[mask] = 2; u_face[mask] = x[mask]/absY[mask]; v_face[mask] = z[mask]/absY[mask]
        mask = (~isYPositive) & (absY >= absX) & (absY >= absZ); face_idx[mask] = 3; u_face[mask] = x[mask]/absY[mask]; v_face[mask] = -z[mask]/absY[mask]
        mask = (isZPositive) & (absZ >= absX) & (absZ >= absY); face_idx[mask] = 4; u_face[mask] = x[mask]/absZ[mask]; v_face[mask] = y[mask]/absZ[mask]
        mask = (~isZPositive) & (absZ >= absX) & (absZ >= absY); face_idx[mask] = 5; u_face[mask] = -x[mask]/absZ[mask]; v_face[mask] = y[mask]/absZ[mask]
        u_face = 0.5 * (u_face + 1.0); v_face = 0.5 * (v_face + 1.0)
        face_size = faces[0].shape[0]
        px_u = np.clip((u_face * (face_size - 1)).astype(int), 0, face_size - 1)
        px_v = np.clip((v_face * (face_size - 1)).astype(int), 0, face_size - 1)
        out_img = np.zeros((out_height, out_width, 3), dtype=np.uint8)
        for i in range(6):
            mask = (face_idx == i)
            if not np.any(mask): continue
            face_data = faces[i]
            rs = px_v[mask]; cs = px_u[mask]
            if face_data.ndim == 3: out_img[mask] = face_data[rs, cs, :3]
            else: vals = face_data[rs, cs]; out_img[mask, 0] = vals; out_img[mask, 1] = vals; out_img[mask, 2] = vals
        return out_img

    @staticmethod
    def get_mip_chain_size(width, height, mip_count, format_code, bit_count):
        total_bytes = 0; w = width; h = height
        block_size = 8 if format_code == b'DXT1' else 16 if format_code in [b'DXT3', b'DXT5'] else 0
        bpp = 4 if block_size == 0 and bit_count != 24 and bit_count != 8 else 3 if bit_count == 24 else 1 if bit_count == 8 else 4
        for _ in range(max(1, mip_count)):
            if block_size > 0: total_bytes += max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * block_size
            else: total_bytes += w * h * bpp
            w = max(1, w // 2); h = max(1, h // 2)
        return total_bytes

    @staticmethod
    def process_dds_to_equi(file_path, target_width=4096):
        print(f"      [Skybox] Processing {os.path.basename(file_path)}")
        faces = []
        try:
            with open(file_path, 'rb') as f:
                header = f.read(128)
                if header[:4] != b'DDS ': return None
                h = struct.unpack_from('<I', header, 12)[0]
                w = struct.unpack_from('<I', header, 16)[0]
                mips = struct.unpack_from('<I', header, 28)[0]
                fourcc = header[84:88]
                bits = struct.unpack_from('<I', header, 88)[0]
                main_bytes = SkyboxProcessor.get_mip_chain_size(w, h, 1, fourcc, bits)
                full_bytes = SkyboxProcessor.get_mip_chain_size(w, h, mips, fourcc, bits)
                skip = full_bytes - main_bytes
                sh = bytearray(header); struct.pack_into('<I', sh, 112, 0); struct.pack_into('<I', sh, 28, 0)
                f.seek(128)
                for _ in range(6):
                    chunk = f.read(main_bytes)
                    if len(chunk) < main_bytes: break
                    if skip > 0: f.seek(skip, 1)
                    try: faces.append(iio.imread(io.BytesIO(sh + chunk), index=0, extension=".dds"))
                    except: return None
            if len(faces) != 6: return None
            return Image.fromarray(SkyboxProcessor.sample_equirectangular(faces, target_width, target_width // 2))
        except Exception as e:
            print(f"      [Skybox] Error: {e}")
            return None

# --- HELPERS ---

def normalize_name(name):
    if not name: return ""
    return name.replace('.lua', '').replace('_', ' ').replace('-', ' ').strip().lower()

def get_lava_data_from_github():
    print("Fetching Lava Level data from GitHub...")
    try:
        response = requests.get(GITHUB_LAVA_URL)
        if response.status_code != 200: return {}
        files = response.json()
        lava_map_data = {}
        level_pattern = re.compile(r'level\s*=\s*(-?\d+(\.\d+)?)')
        for file in files:
            if file['name'].endswith('.lua'):
                raw = requests.get(file['download_url']).text
                clean = normalize_name(file['name'])
                match = level_pattern.search(raw)
                lava_map_data[clean] = int(float(match.group(1))) if match else 0
        return lava_map_data
    except Exception: return {}

def upload_to_ftp(local_path, target_filename):
    if not all([FTP_HOST, FTP_USER, FTP_PASSWORD, FTP_PATH, FTP_BASE_URL]): return None
    try:
        with ftplib.FTP(host=FTP_HOST, timeout=60) as ftp:
            ftp.login(user=FTP_USER, passwd=FTP_PASSWORD)
            try: ftp.cwd(FTP_PATH)
            except: return None
            with open(local_path, 'rb') as f:
                ftp.storbinary(f'STOR {target_filename}', f)
        return f"{FTP_BASE_URL.rstrip('/')}/{target_filename}"
    except Exception as e:
        print(f"   -> FTP Error: {e}")
        return None

def delete_from_ftp(target_filenames):
    if not all([FTP_HOST, FTP_USER, FTP_PASSWORD, FTP_PATH]): return
    try:
        with ftplib.FTP(host=FTP_HOST, timeout=30) as ftp:
            ftp.login(user=FTP_USER, passwd=FTP_PASSWORD)
            ftp.cwd(FTP_PATH)
            for fname in target_filenames:
                if fname:
                    try: ftp.delete(fname)
                    except: pass
    except: pass

def save_and_upload_pil_image(img, item_id, suffix):
    filename = f"{item_id}_{suffix}.webp"
    if img.width > MAX_PIXEL_DIMENSION or img.height > MAX_PIXEL_DIMENSION:
        print(f"      [Image Processing] Initial resize from {img.width}x{img.height} to max {MAX_PIXEL_DIMENSION}px")
        img.thumbnail((MAX_PIXEL_DIMENSION, MAX_PIXEL_DIMENSION), Image.Resampling.LANCZOS)
    
    use_lossless = suffix in ["height", "metal"]
    quality = 85
    current_img = img 
    resize_factor = 1.0
    
    while True:
        buf = io.BytesIO()
        if resize_factor < 1.0:
            new_w = int(img.width * resize_factor)
            new_h = int(img.height * resize_factor)
            current_img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        
        current_img.save(buf, format="WEBP", quality=quality, lossless=use_lossless)
        size_mb = buf.tell() / (1024 * 1024)
        print(f"      [Image Processing] {suffix}: {current_img.width}x{current_img.height} = {size_mb:.2f} MB")

        if size_mb <= MAX_FILE_SIZE_MB:
            with open(filename, "wb") as f:
                f.write(buf.getbuffer())
            break
        
        if use_lossless: resize_factor *= 0.75 
        else:
            if quality > 50: quality -= 10 
            else: resize_factor *= 0.75 
        if resize_factor < 0.1: return None

    print(f"      [FTP] Uploading {filename} ({size_mb:.2f} MB)...")
    url = upload_to_ftp(filename, filename)
    if url:
        if os.path.exists(filename): os.remove(filename)
        return url
    return None

def process_texture_from_archive(seven_zip_file, texture_filename, item_id, suffix, temp_dir):
    if not texture_filename: return None, None
    all_files = seven_zip_file.getnames()
    target_file = next((f for f in all_files if f.lower().endswith(texture_filename.lower())), None)
    if not target_file: return None, None

    seven_zip_file.extract(targets=[target_file], path=temp_dir)
    raw_path = os.path.join(temp_dir, target_file)
    
    if suffix == "sky":
        img_pil = SkyboxProcessor.process_dds_to_equi(raw_path, target_width=SKYBOX_MAX_WIDTH)
        if img_pil:
            public_url = save_and_upload_pil_image(img_pil, item_id, "sky")
            return public_url, f"{item_id}_sky.webp"
    else:
        try:
            img = Image.open(raw_path)
            img.load()
            if suffix == "normal":
                extrema = img.convert("L").getextrema()
                if extrema[1] == 0: 
                    print(f"      [Image] Skipped {texture_filename}: Image is completely black.")
                    return None, None
            public_url = save_and_upload_pil_image(img, item_id, suffix)
            return public_url, f"{item_id}_{suffix}.webp"
        except Exception as e:
            print(f"      [Image] Error opening {texture_filename}: {e}")
    return None, None

def get_maps_to_process(new_only=False, map_filter=None):
    url = f"https://api.webflow.com/v2/collections/{COLLECTION_ID}/items"
    items_to_process = []
    offset = 0; limit = 100
    
    if map_filter:
        print(f"Checking Webflow items (FILTER: \"{map_filter}\")...")
    elif new_only:
        print("Checking Webflow items (NEW-ONLY MODE: skipping maps that already have a version)...")
    else:
        print("Checking Webflow items...")
    while True:
        params = {'limit': limit, 'offset': offset}
        try:
            response = requests.get(url, headers=HEADERS_WEBFLOW, params=params)
            response.raise_for_status()
            data = response.json()
        except: break

        current_batch = data.get('items', [])
        if not current_batch: break

        for item in current_batch:
            fields = item.get('fieldData', {})
            
            # MAP FILTER: skip maps that don't match the search term
            if map_filter:
                item_name = fields.get('name', '')
                if map_filter.lower() not in item_name.lower():
                    continue
                # For filtered maps, force-update everything
                item['tasks'] = {
                    "diffuse": True, "height": True, "metal": True,
                    "normal": True, "skybox": True,
                    "water_tint": True, "water_base": True,
                    "water_min": True, "water_absorb": True,
                    "minmax": True
                }
                item['current_version'] = fields.get(FIELD_VERSION)
                items_to_process.append(item)
                continue
            
            webflow_version = fields.get(FIELD_VERSION)
            
            # NEW-ONLY MODE: skip any map that already has a version in Webflow
            if new_only:
                if webflow_version:
                    continue
                # For new maps, treat everything as missing
                item['tasks'] = {
                    "diffuse": True, "height": True, "metal": True,
                    "normal": True, "skybox": True,
                    "water_tint": True, "water_base": True,
                    "water_min": True, "water_absorb": True,
                    "minmax": True
                }
            else:
                missing_texture = fields.get(FIELD_TEXTURE_MAP) is None
                missing_height = fields.get(FIELD_HEIGHT_MAP) is None
                missing_metal = fields.get(FIELD_METAL_MAP) is None
                missing_normal = fields.get(FIELD_NORMAL_MAP) is None
                missing_skybox = fields.get(FIELD_SKYBOX) is None
                missing_tint = fields.get(FIELD_WATER_TINT) is None
                missing_base = fields.get(FIELD_WATER_BASE) is None
                missing_min = fields.get(FIELD_WATER_MIN) is None
                missing_absorb = fields.get(FIELD_WATER_ABSORB) is None
                missing_minmax = fields.get(FIELD_MIN) is None
                
                item['tasks'] = {
                    "diffuse": FORCE_CORE_OVERWRITE or missing_texture,
                    "height": FORCE_CORE_OVERWRITE or missing_height,
                    "metal": FORCE_CORE_OVERWRITE or missing_metal,
                    "normal": FORCE_HEAVY_OVERWRITE or missing_normal,
                    "skybox": FORCE_HEAVY_OVERWRITE or missing_skybox,
                    "water_tint": FORCE_METADATA_UPDATE or missing_tint,
                    "water_base": FORCE_METADATA_UPDATE or missing_base,
                    "water_min": FORCE_METADATA_UPDATE or missing_min,
                    "water_absorb": FORCE_METADATA_UPDATE or missing_absorb,
                    "minmax": FORCE_METADATA_UPDATE or missing_minmax
                }
            
            item['current_version'] = webflow_version
            items_to_process.append(item)
        
        if len(current_batch) < limit: break
        offset += limit
    return items_to_process

def extract_map_data(sd7_url, item_id, tasks, current_webflow_version):
    if not sd7_url: 
        return {"min": None, "max": None, "void": False, "normal_url": None, 
                "skybox_url": None, "texture_url": None, "height_url": None, "metal_url": None,
                "water_tint": None, "water_base": None, "water_min": None, "water_absorb": None, 
                "version": None, "cleanup_list": []}

    temp_archive = "temp_map.sd7"; temp_extract_dir = "temp_extract"
    result = {"min": None, "max": None, "void": False, 
              "normal_url": None, "skybox_url": None,
              "texture_url": None, "height_url": None, "metal_url": None,
              "water_tint": None, "water_base": None, "water_min": None, "water_absorb": None, 
              "version": None, "cleanup_list": []}
    
    if os.path.exists(temp_extract_dir): shutil.rmtree(temp_extract_dir)
    if os.path.exists(temp_archive): os.remove(temp_archive)
    
    try:
        print(f"   -> Downloading SD7: {sd7_url}")
        with requests.get(sd7_url, stream=True) as r:
            r.raise_for_status()
            with open(temp_archive, 'wb') as f: shutil.copyfileobj(r.raw, f)
        
        if py7zr.is_7zfile(temp_archive):
            with py7zr.SevenZipFile(temp_archive, mode='r') as z:
                all_files = z.getnames()
                mapinfo_file = next((f for f in all_files if f.lower() == "mapinfo.lua"), None)
                smt_hint = None
                
                if mapinfo_file:
                    z.extract(targets=[mapinfo_file], path=temp_extract_dir)
                    with open(os.path.join(temp_extract_dir, mapinfo_file), 'r', encoding='utf-8', errors='ignore') as f:
                        raw_content = f.read()
                        
                        # --- VERSIE CHECK: gebruik ruwe content (voor extract_version_from_lua) ---
                        found_version = extract_version_from_lua(raw_content)
                        
                        # --- STRIP COMMENTS voor alle andere verwerking ---
                        content = strip_lua_comments(raw_content)
                        
                        if found_version:
                            result["version"] = found_version
                            print(f"      [MapInfo] Found Version: {found_version}")
                            
                            is_forced = FORCE_VERSION_OVERWRITE
                            is_changed = (found_version != current_webflow_version)
                            
                            if is_changed or is_forced:
                                reason = "Version Changed" if is_changed else "Forced Overwrite"
                                print(f"      [UPDATE DETECTED] {reason} (Webflow: {current_webflow_version} -> New: {found_version}). FORCING FULL REFRESH.")
                                for key in tasks:
                                    tasks[key] = True
                        
                        if tasks["minmax"]:
                            min_match = re.search(r'minheight\s*=\s*([\d\.-]+)', content, re.IGNORECASE)
                            max_match = re.search(r'maxheight\s*=\s*([\d\.-]+)', content, re.IGNORECASE)
                            if min_match: result["min"] = float(min_match.group(1))
                            if max_match: result["max"] = float(max_match.group(1))
                        
                        if re.search(r'voidWater\s*=\s*(true|1)', content, re.IGNORECASE): result["void"] = True
                        
                        if tasks["water_tint"]:
                            hex_color = extract_lua_value(content, "water", ["surfaceColor", "diffuseColor"])
                            if hex_color:
                                result["water_tint"] = rgb_float_to_hex(hex_color)
                                print(f"      [MapInfo] Found Water Tint: {result['water_tint']}")

                        if tasks["water_base"]:
                            hex_color = extract_lua_value(content, "water", ["baseColor"])
                            if hex_color:
                                result["water_base"] = rgb_float_to_hex(hex_color)
                                print(f"      [MapInfo] Found Water Base: {result['water_base']}")

                        if tasks["water_min"]:
                            hex_color = extract_lua_value(content, "water", ["mincolor"])
                            if hex_color:
                                result["water_min"] = rgb_float_to_hex(hex_color)
                                print(f"      [MapInfo] Found Water Min: {result['water_min']}")

                        if tasks["water_absorb"]:
                            raw_absorb = extract_lua_value(content, "water", ["absorb"])
                            if raw_absorb:
                                result["water_absorb"] = raw_absorb
                                print(f"      [MapInfo] Found Water Absorb: {result['water_absorb']}")

                        if tasks["normal"]:
                            nm_match = re.search(r'detailNormalTex\s*=\s*["\']([^"\']+)["\']', content, re.IGNORECASE)
                            if nm_match: 
                                url, fname = process_texture_from_archive(z, nm_match.group(1), item_id, "normal", temp_extract_dir)
                                result["normal_url"] = url
                                if fname: result["cleanup_list"].append(fname)
                        
                        if tasks["skybox"]:
                            sb_match = re.search(r'skyBox\s*=\s*["\']([^"\']+)["\']', content, re.IGNORECASE)
                            if sb_match:
                                url, fname = process_texture_from_archive(z, sb_match.group(1), item_id, "sky", temp_extract_dir)
                                result["skybox_url"] = url
                                if fname: result["cleanup_list"].append(fname)
                        
                        smt_hint = SpringMapParser.get_filenames_from_lua(content)
                
                if tasks["diffuse"] or tasks["height"] or tasks["metal"]:
                    target_map_files = [f for f in all_files if f.lower().endswith('.smf')]
                    if tasks["diffuse"]:
                        target_map_files += [f for f in all_files if f.lower().endswith('.smt')]
                        
                    if target_map_files:
                        print(f"      [MapParser] Extracting required files...")
                        z.extract(targets=target_map_files, path=temp_extract_dir)
                        map_images = SpringMapParser.process_map_files(temp_extract_dir, mapinfo_file, tasks, smt_hint)
                        
                        if "diffuse" in map_images:
                            url = save_and_upload_pil_image(map_images["diffuse"], item_id, "texture")
                            result["texture_url"] = url
                            if url: result["cleanup_list"].append(f"{item_id}_texture.webp")
                        if "heightmap" in map_images:
                            url = save_and_upload_pil_image(map_images["heightmap"], item_id, "height")
                            result["height_url"] = url
                            if url: result["cleanup_list"].append(f"{item_id}_height.webp")
                        if "metalmap" in map_images:
                            url = save_and_upload_pil_image(map_images["metalmap"], item_id, "metal")
                            result["metal_url"] = url
                            if url: result["cleanup_list"].append(f"{item_id}_metal.webp")

    except Exception as e:
        print(f"   -> SD7 Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(temp_extract_dir): shutil.rmtree(temp_extract_dir)
        if os.path.exists(temp_archive): os.remove(temp_archive)

    return result

def update_webflow_item(item_id, data, lava_level=None):
    url_update = f"https://api.webflow.com/v2/collections/{COLLECTION_ID}/items/{item_id}"
    fields = {}

    if data["min"] is not None: fields[FIELD_MIN] = data["min"]
    if data["max"] is not None: fields[FIELD_MAX] = data["max"]
    fields[FIELD_VOID_WATER] = data["void"]
    if data["normal_url"]: fields[FIELD_NORMAL_MAP] = data["normal_url"]
    if data["water_tint"]: fields[FIELD_WATER_TINT] = data["water_tint"]
    if data["water_base"]: fields[FIELD_WATER_BASE] = data["water_base"]
    if data["water_min"]: fields[FIELD_WATER_MIN] = data["water_min"]
    if data["water_absorb"]: fields[FIELD_WATER_ABSORB] = data["water_absorb"]
    if data["version"]: fields[FIELD_VERSION] = data["version"]
    
    if data["skybox_url"]: fields[FIELD_SKYBOX] = data["skybox_url"]
    if data["texture_url"]: fields[FIELD_TEXTURE_MAP] = data["texture_url"]
    if data["height_url"]: fields[FIELD_HEIGHT_MAP] = data["height_url"]
    if data["metal_url"]: fields[FIELD_METAL_MAP] = data["metal_url"]
    
    if lava_level is not None: fields[FIELD_LAVA_LEVEL] = lava_level

    if not fields: return False

    try:
        response = requests.patch(url_update, json={"fieldData": fields}, headers=HEADERS_WEBFLOW)
        if response.status_code == 200:
            print(f"   -> Webflow Updated (STAGED CHANGES - NOT PUBLISHED)")
            print(f"      [DEBUG] Updated fields: {', '.join(fields.keys())}")
            return True
        else:
            print(f"   -> UPDATE FAILED: {response.text}")
            return False
    except Exception as e:
        print(f"   -> API Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="BAR Map Sync — sync map data from SD7 archives to Webflow CMS")
    parser.add_argument('--new-only', action='store_true',
                        help='Only process maps that have no version in Webflow yet (new/empty maps)')
    parser.add_argument('--map', type=str, default=None,
                        help='Process only a specific map by name (case-insensitive partial match, e.g. --map "Throne")')
    args = parser.parse_args()

    lava_lookup = get_lava_data_from_github()
    items = get_maps_to_process(new_only=args.new_only, map_filter=args.map)
    
    if not items:
        if args.map:
            print(f"No maps found matching \"{args.map}\".")
        elif args.new_only:
            print("No NEW maps found (all maps already have a version). Nothing to do.")
        else:
            print("No maps to process.")
        return

    mode_label = f"SINGLE MAP: {args.map}" if args.map else "NEW-ONLY MODE" if args.new_only else "VERSION SYNC MODE"
    print(f"--- Processing {len(items)} maps ({mode_label}) ---\n")
    
    for item in items:
        name = item['fieldData'].get('name', 'Nameless')
        tasks = item.get('tasks', {})
        current_version = item.get('current_version')
        print(f"Processing: {name} (Current: {current_version})")
        
        todo = [k for k,v in tasks.items() if v]
        print(f"   -> Missing/Updating: {', '.join(todo)}")

        clean_name = normalize_name(name)
        found_lava_level = lava_lookup.get(clean_name)
        url = item['fieldData'].get(FIELD_DOWNLOAD_URL)
        
        sd7_data = extract_map_data(url, item['id'], tasks, current_version)
        
        has_updates = (
            sd7_data["min"] is not None or 
            sd7_data["water_tint"] or 
            sd7_data["water_base"] or 
            sd7_data["water_min"] or 
            sd7_data["water_absorb"] or 
            sd7_data["texture_url"] or 
            sd7_data["height_url"] or 
            sd7_data["metal_url"] or 
            sd7_data["normal_url"] or 
            sd7_data["skybox_url"] or 
            sd7_data["version"] or
            found_lava_level is not None
        )

        if has_updates:
            if update_webflow_item(item['id'], sd7_data, found_lava_level):
                if sd7_data["cleanup_list"]:
                    time.sleep(1)
                    delete_from_ftp(sd7_data["cleanup_list"])
            time.sleep(1)
        else:
            print("   -> Skip: No new data or version update.")

if __name__ == "__main__":
    main()