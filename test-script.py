import struct
import os
import re
import py7zr
import requests

# --- DIAGNOSE SCRIPT ---
# Paste the URL of the failing map below (e.g. Painted Desert or Aethermoor)
TEST_URL = "https://files-cdn.beyondallreason.dev/file/0af9c293ffcbab453a4ddb0ebf1cf4fb/aethermoor_creek_1.0.sd7"

def analyze_header():
    print(f"Downloading {TEST_URL.split('/')[-1]}...")
    r = requests.get(TEST_URL)
    with open("debug_map.sd7", "wb") as f:
        f.write(r.content)
        
    print("Extracting SMF header...")
    with py7zr.SevenZipFile("debug_map.sd7", 'r') as z:
        for fname in z.getnames():
            if fname.endswith(".smf"):
                print(f"Found: {fname}")
                z.extract(targets=[fname])
                
                with open(fname, "rb") as f_smf:
                    # Read the first 100 bytes as integers
                    f_smf.seek(0)
                    data = f_smf.read(100)
                    ints = struct.unpack('<25I', data)
                    
                    print("\n--- HEADER DUMP (Ints) ---")
                    labels = [
                        "0: Magic part 1", "1: Magic part 2", "2: Magic part 3", "3: Magic part 4",
                        "4: Version", "5: MapID", "6: MapWidth", "7: MapHeight",
                        "8: SquareSize", "9: TexelsPerSquare", "10: TileSize",
                        "11: MinDepth", "12: MaxDepth", 
                        "13: HeightMapPtr", "14: TypeMapPtr", "15: TileIndexMapPtr",
                        "16: MiniMapPtr", "17: MetalMapPtr", "18: FeaturePtr",
                        "19: ExtraPtr1", "20: ExtraPtr2", "21: ExtraPtr3"
                    ]
                    
                    for i, val in enumerate(ints):
                        if i < len(labels):
                            print(f"Index {i:02d} | {labels[i]:<20} : {val}")
                        else:
                            print(f"Index {i:02d} | {'...':<20} : {val}")
                            
    if os.path.exists("debug_map.sd7"): os.remove("debug_map.sd7")
    print("\nLook for where the LARGE numbers start (those are the pointers).")

if __name__ == "__main__":
    analyze_header()