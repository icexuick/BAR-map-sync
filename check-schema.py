import requests
import os
import json
from dotenv import load_dotenv

# Make sure your .env file is in the same directory
load_dotenv()

WEBFLOW_API_TOKEN = os.environ.get("WEBFLOW_API_TOKEN")
# Collection ID from the main script
COLLECTION_ID = "6564c6553676389f8ba45aaf"

HEADERS = {
    "Authorization": f"Bearer {WEBFLOW_API_TOKEN}",
    "accept-version": "2.0.0",
    "content-type": "application/json"
}

def get_collection_schema():
    url = f"https://api.webflow.com/v2/collections/{COLLECTION_ID}"
    
    try:
        print(f"Fetching schema for collection: {COLLECTION_ID}...")
        response = requests.get(url, headers=HEADERS)

        if response.status_code == 200:
            data = response.json()
            fields = data.get('fields', [])

            print("\n--- Fields found ---")
            print(f"{'LABEL':<30} | {'SLUG (use this in script)':<30} | {'TYPE'}")
            print("-" * 80)

            found_texture = False
            for field in fields:
                label = field.get('displayName', 'N/A')
                slug = field.get('slug', 'N/A')
                f_type = field.get('type', 'N/A')

                print(f"{label:<30} | {slug:<30} | {f_type}")

                if "texture" in slug.lower():
                    found_texture = True

            print("-" * 80)

            if not found_texture:
                print("\nWARNING: no field with 'texture' in the slug.")
                print("Check whether the field was created or perhaps named 'Image'.")
            else:
                print("\nTexture field found. Copy the exact slug above into your main script.")

        else:
            print(f"Fetch failed: {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    get_collection_schema()