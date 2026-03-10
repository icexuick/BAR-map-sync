# BAR Map Sync — Beyond All Reason Map Data Pipeline

Syncs map data (height values, textures, metadata, skyboxes) from Beyond All Reason `.sd7` map archives to the Webflow CMS collection used by [beyondallreason.info](https://beyondallreason.info).

---

## Overview

This project contains **two main sync scripts** and several utility/debug scripts:

| Script | Purpose | Runs on |
|---|---|---|
| `update_maps_local.py` | **Full sync** — extracts textures, heightmaps, metalmaps, skyboxes, water metadata, lava levels, and version info. Uploads images to FTP and updates Webflow. | Your local machine |
| `update_maps.py` | **Lightweight sync** — fills in only missing `minheight` / `maxheight` / `voidWater` values. Auto-publishes items. | GitHub Actions (or local) |
| `check-schema.py` | Prints all Webflow collection field slugs and types. Useful for verifying field names before editing sync code. | Local |
| `test-script.py` | Debug tool — downloads one map and dumps the raw SMF header bytes for analysis. | Local |
| `map-desert-checker.py` | Debug tool — hex-dumps the first 64 bytes of a specific map's SMF file. | Local |

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/bar-map-sync.git
cd bar-map-sync
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

The local script additionally needs these packages (not in `requirements.txt`):

```bash
pip install python-dotenv numpy imageio pillow
```

> **Tip:** `imageio` is used for DDS cubemap face decoding (skybox processing).

### 3. Create a `.env` file

Create a file called `.env` in the project root (it's in `.gitignore` so it won't be committed):

```env
WEBFLOW_API_TOKEN=your_webflow_api_token_here

# FTP credentials (needed for texture/skybox uploads in update_maps_local.py)
FTP_HOST=ftp.example.com
FTP_USER=your_ftp_username
FTP_PASSWORD=your_ftp_password
FTP_PATH=/public_html/maps
FTP_BASE_URL=https://example.com/maps
```

### 4. GitHub Actions (automated lightweight sync)

The workflow at `.github/workflows/update_maps.yml` runs `update_maps.py` every hour. It only needs one secret:

- Go to your repo → **Settings → Secrets → Actions**
- Add `WEBFLOW_API_TOKEN` as a repository secret

The workflow also has a **Run workflow** button for manual triggers.

---

## Scripts in Detail

### `update_maps_local.py` — Full Local Sync

This is the main workhorse. It processes **every map** in the Webflow collection and can extract/update all data fields.

#### How to run

```bash
# Standard run (uses flag constants in the script)
python update_maps_local.py

# NEW-ONLY MODE: only process maps that have no version set yet in Webflow
python update_maps_local.py --new-only
```

#### Command-line arguments

| Argument | Description |
|---|---|
| `--new-only` | Only process maps where the `version` field in Webflow is empty. Skips all maps that already have data. Great for quickly picking up newly added maps without re-processing the entire collection. |

#### Configuration flags (in the script)

When running **without** `--new-only`, behavior is controlled by these constants at the top of the file:

| Flag | Default | What it does |
|---|---|---|
| `FORCE_VERSION_OVERWRITE` | `True` | Always check the version field in `mapinfo.lua`. If the version has changed (or this is `True`), it forces a full refresh of all fields for that map. |
| `FORCE_CORE_OVERWRITE` | `True` | Force re-extraction of **diffuse texture**, **heightmap**, and **metalmap** even if they already exist in Webflow. |
| `FORCE_HEAVY_OVERWRITE` | `False` | Force re-extraction of **normal map** and **skybox** even if already present. These are the slowest/largest operations. |
| `FORCE_METADATA_UPDATE` | `True` | Force refresh of **water colors** (tint, base, min, absorb) and **min/max height**. |

#### Typical use cases

**"I want to update everything from scratch":**
Set all `FORCE_*` flags to `True`. This will re-download and re-process every map.

**"I only want to fill in missing data":**
Set all `FORCE_*` flags to `False`. The script will skip any field that already has a value in Webflow.

**"New maps were added to the collection, I want to sync only those":**
```bash
python update_maps_local.py --new-only
```
This skips all maps that already have a `version` value in Webflow, so only brand-new (empty) maps get processed. All `FORCE_*` flags are ignored — every field is treated as missing.

**"A few maps got updated, I want to refresh only those":**
Leave `FORCE_VERSION_OVERWRITE = True` (default). The script compares the `version` field from `mapinfo.lua` against what's stored in Webflow. If it changed, all fields for that map are refreshed.

**"I only want to re-process skyboxes and normal maps":**
Set `FORCE_HEAVY_OVERWRITE = True` and the rest to `False`.

#### What data gets extracted

From each `.sd7` map archive, the script extracts:

| Data | Source | Webflow field slug |
|---|---|---|
| Min/Max height | `mapinfo.lua` → `minheight` / `maxheight` | `map-height-min`, `map-height-max` |
| Void water | `mapinfo.lua` → `voidWater = true` | `void-water` |
| Version | `mapinfo.lua` → `version = "..."` | `version` |
| Water tint color | `mapinfo.lua` → `water { surfaceColor / diffuseColor }` | `water-lava-color-tint` |
| Water base color | `mapinfo.lua` → `water { baseColor }` | `water-basecolor` |
| Water min color | `mapinfo.lua` → `water { mincolor }` | `water-min` |
| Water absorb | `mapinfo.lua` → `water { absorb }` | `water-absorb` |
| Normal map | `mapinfo.lua` → `detailNormalTex` reference → extract from archive | `normal-map` |
| Skybox | `mapinfo.lua` → `skyBox` reference → DDS cubemap → equirectangular projection | `skybox` |
| Diffuse texture | `.smf` + `.smt` files → DXT1 tile stitching | `mini-map` |
| Heightmap | `.smf` file → raw uint16 data → 8-bit grayscale | `height-map` |
| Metalmap | `.smf` file → raw uint8 data | `metal-map` |
| Lava level | GitHub `LavaMaps` config files | `lavalevel` |

#### Image processing details

- All images are saved as **WebP** format
- Maximum dimension: **4096×4096** pixels
- Maximum file size: **4 MB** (auto-compressed/downscaled if exceeded)
- Heightmaps and metalmaps use **lossless** compression
- Textures use **lossy** compression (quality 85, reduced if over 4MB)
- Skybox DDS cubemaps are converted to **equirectangular projection** (4096×2048)
- Images are uploaded to **FTP**, and the public URL is stored in Webflow

#### Important notes

- Changes are **staged** in Webflow (not auto-published). You need to publish manually from the Webflow CMS dashboard or add a publish API call.
- The script writes temporary files (`temp_map.sd7`, `temp_extract/`) in the working directory and cleans them up after each map.
- Lava level data is fetched from the BAR GitHub repo (`common/configs/LavaMaps/`).

---

### `update_maps.py` — Lightweight GitHub Actions Sync

A simpler script that only handles `minheight`, `maxheight`, and `voidWater`. It **auto-publishes** each item after updating.

#### How to run

```bash
# Locally (needs WEBFLOW_API_TOKEN in env or .env)
export WEBFLOW_API_TOKEN=your_token_here
python update_maps.py

# Or via GitHub Actions (automatic, runs hourly)
```

#### Behavior

1. Fetches all items from the Webflow collection
2. Filters to only items where `map-height-min` OR `map-height-max` is empty
3. Downloads each map's `.sd7`, extracts `mapinfo.lua`
4. Parses `minheight`, `maxheight`, and `voidWater`
5. Updates the Webflow item **and publishes it immediately**

This script has no configuration flags or arguments — it always processes only items with missing height data.

---

### `check-schema.py` — Webflow Collection Field Inspector

Prints a formatted table of all fields in the Maps collection, including their display name, slug, and type.

```bash
python check-schema.py
```

**Example output:**
```
--- Gevonden Velden ---
LABEL                          | SLUG (Gebruik deze in script)  | TYPE
--------------------------------------------------------------------------------
Name                           | name                           | PlainText
Download URL                   | downloadurl                    | Link
Map Height Min                 | map-height-min                 | Number
...
```

Use this to verify the exact slug of a field before adding it to the sync scripts.

---

### `test-script.py` — SMF Header Debugger

Downloads a specific map and dumps the SMF binary header as labeled integers. Edit the `TEST_URL` variable at the top to point to the map you want to inspect.

```bash
# Edit TEST_URL in the file first, then:
python test-script.py
```

### `map-desert-checker.py` — Hex Dump Debugger

Similar to `test-script.py` but does a raw hex dump of the first 64 bytes. Hardcoded to a specific map URL (edit `MAP_URL` to change).

```bash
python map-desert-checker.py
```

---

## GitHub Actions Workflow

**File:** `.github/workflows/update_maps.yml`

| Trigger | Description |
|---|---|
| `schedule: cron '0 * * * *'` | Runs automatically every hour |
| `workflow_dispatch` | Manual trigger via GitHub UI (Actions tab → Run workflow) |

The workflow runs `update_maps.py` (lightweight mode) with Python 3.9.

### Required secret

| Secret name | Where to set it |
|---|---|
| `WEBFLOW_API_TOKEN` | Repo → Settings → Secrets and variables → Actions |

---

## Environment Variables Reference

| Variable | Required by | Description |
|---|---|---|
| `WEBFLOW_API_TOKEN` | All scripts | Webflow API v2 bearer token |
| `FTP_HOST` | `update_maps_local.py` | FTP server hostname |
| `FTP_USER` | `update_maps_local.py` | FTP username |
| `FTP_PASSWORD` | `update_maps_local.py` | FTP password |
| `FTP_PATH` | `update_maps_local.py` | Remote directory path on FTP server |
| `FTP_BASE_URL` | `update_maps_local.py` | Public URL prefix for uploaded files |

---

## Webflow Collection

- **Collection ID:** `6564c6553676389f8ba45aaf`
- All field slugs are defined as constants at the top of each script (e.g. `FIELD_MIN`, `FIELD_TEXTURE_MAP`, etc.)

---

## Quick Reference

```bash
# Full sync (all maps, respects FORCE_* flags)
python update_maps_local.py

# Only process NEW maps (no version in Webflow yet)
python update_maps_local.py --new-only

# Lightweight sync (missing heights only, auto-publishes)
python update_maps.py

# Check what fields exist in the Webflow collection
python check-schema.py
```

---

## File Structure

```
bar-map-sync/
├── .env                          # Your secrets (not committed)
├── .gitignore                    # Ignores .env
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── update_maps_local.py          # Full sync (local, all data)
├── update_maps.py                # Lightweight sync (heights only)
├── check-schema.py               # Print Webflow field slugs
├── test-script.py                # Debug: SMF header dump
├── map-desert-checker.py         # Debug: hex dump
└── .github/
    └── workflows/
        └── update_maps.yml       # GitHub Actions workflow
```
