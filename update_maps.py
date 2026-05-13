import requests
import py7zr
import re
import os
import shutil
import time

# --- CONFIGURATION ---
WEBFLOW_API_TOKEN = os.environ.get("WEBFLOW_API_TOKEN")
if not WEBFLOW_API_TOKEN:
    raise ValueError("CRITICAL: No WEBFLOW_API_TOKEN found in environment variables.")

COLLECTION_ID = "6564c6553676389f8ba45aaf"
FIELD_MIN = "map-height-min"
FIELD_MAX = "map-height-max"
FIELD_DOWNLOAD_URL = "downloadurl"
FIELD_VOID_WATER = "void-water" 

HEADERS = {
    "Authorization": f"Bearer {WEBFLOW_API_TOKEN}",
    "accept-version": "2.0.0",
    "content-type": "application/json"
}

def get_maps_without_height():
    """
    Fetches items from Webflow.
    Filters: Only keeps items where min OR max height is missing (None).
    """
    url = f"https://api.webflow.com/v2/collections/{COLLECTION_ID}/items"
    items_to_process = []
    offset = 0
    limit = 100
    
    print("Checking for NEW maps (missing height data)...")
    
    while True:
        params = {'limit': limit, 'offset': offset}
        try:
            response = requests.get(url, headers=HEADERS, params=params)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Error fetching items: {e}")
            break

        current_batch = data.get('items', [])
        if not current_batch:
            break

        # Check per item whether the data is already present
        for item in current_batch:
            fields = item.get('fieldData', {})

            # --- THE FILTER ---
            # If min or max height is empty, add it to the to-do list.
            # Already populated? Then ignore it.
            if fields.get(FIELD_MIN) is None or fields.get(FIELD_MAX) is None:
                items_to_process.append(item)
        
        if len(current_batch) < limit:
            break
            
        offset += limit
            
    return items_to_process

def extract_map_info(sd7_url):
    """
    Downloads the map, extracts ONLY the mapinfo.lua in the root, 
    parses min/max height AND voidWater.
    """
    temp_archive = "temp_map.sd7"
    temp_extract_dir = "temp_extract"
    min_h, max_h = None, None
    void_water = False 
    
    # Cleanup beforehand
    if os.path.exists(temp_extract_dir):
        shutil.rmtree(temp_extract_dir)
    if os.path.exists(temp_archive):
        os.remove(temp_archive)
    
    try:
        # 1. Download
        print(f"   -> Downloading: {sd7_url}")
        with requests.get(sd7_url, stream=True) as r:
            r.raise_for_status()
            with open(temp_archive, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        
        # 2. Open archive
        if py7zr.is_7zfile(temp_archive):
            with py7zr.SevenZipFile(temp_archive, mode='r') as z:
                all_files = z.getnames()
                
                target_file = None
                for f in all_files:
                    if f.lower() == "mapinfo.lua":
                        target_file = f
                        break
                
                if not target_file:
                    print("   -> NO mapinfo.lua found in root (subfolders ignored).")
                    return None, None, False
                
                # 3. Extract (only that single file)
                z.extract(targets=[target_file], path=temp_extract_dir)
                local_path = os.path.join(temp_extract_dir, target_file)
                
                # 4. Read and Parse
                with open(local_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                    # Regex for Height
                    min_match = re.search(r'minheight\s*=\s*([\d\.-]+)', content, re.IGNORECASE)
                    max_match = re.search(r'maxheight\s*=\s*([\d\.-]+)', content, re.IGNORECASE)
                    
                    if min_match:
                        min_h = float(min_match.group(1))
                    if max_match:
                        max_h = float(max_match.group(1))

                    # Regex for VoidWater
                    void_match = re.search(r'voidWater\s*=\s*(true|1)', content, re.IGNORECASE)
                    if void_match:
                        void_water = True

    except Exception as e:
        print(f"   -> Error processing map: {e}")
    finally:
        # Cleanup afterwards
        if os.path.exists(temp_extract_dir):
            shutil.rmtree(temp_extract_dir)
        if os.path.exists(temp_archive):
            os.remove(temp_archive)

    return min_h, max_h, void_water

def update_webflow_item(item_id, min_h, max_h, void_water):
    """Sends data to Webflow AND publishes the item immediately."""
    
    url_update = f"https://api.webflow.com/v2/collections/{COLLECTION_ID}/items/{item_id}"
    
    payload = {
        "fieldData": {
            FIELD_MIN: min_h,
            FIELD_MAX: max_h,
            FIELD_VOID_WATER: void_water
        }
    }
    
    try:
        # 1. Update request
        response = requests.patch(url_update, json=payload, headers=HEADERS)
        if response.status_code == 200:
            print(f"   -> Update successful (staged: H:{min_h}/{max_h} Void:{void_water}). Now publishing...")
            
            # STEP 2: Publish this specific item
            url_publish = f"https://api.webflow.com/v2/collections/{COLLECTION_ID}/items/publish"
            payload_publish = {
                "itemIds": [item_id]
            }
            
            pub_response = requests.post(url_publish, json=payload_publish, headers=HEADERS)
            
            if pub_response.status_code in [200, 202]:
                print(f"   -> SUCCESS: Item is LIVE!")
            else:
                print(f"   -> ERROR publishing: {pub_response.text}")
                
        else:
            print(f"   -> UPDATE FAILED: {response.text}")

    except Exception as e:
        print(f"   -> API Error: {e}")

def main():
    # Only fetch the new maps
    items = get_maps_without_height()
    
    if not items:
        print("No new maps to process. Exiting.")
        return

    print(f"--- Start processing {len(items)} NEW maps ---\n")
    
    for item in items:
        name = item['fieldData'].get('name', 'Nameless')
        url = item['fieldData'].get(FIELD_DOWNLOAD_URL)
        
        if not url:
            print(f"Skipping: {name} (No download URL)")
            continue
            
        print(f"Processing: {name}")
        min_h, max_h, void_water = extract_map_info(url)
        
        if min_h is not None and max_h is not None:
            print(f"   -> Found: Min {min_h} / Max {max_h} / VoidWater: {void_water}")
            update_webflow_item(item['id'], min_h, max_h, void_water)
        else:
            print("   -> No usable height data found.")

        # Respect API rate limits
        time.sleep(2) 

if __name__ == "__main__":
    main()