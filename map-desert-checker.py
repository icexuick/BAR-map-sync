import struct
import py7zr
import requests
import os
import shutil

# De URL van de probleem-map
MAP_URL = "https://files-cdn.beyondallreason.dev/file/e701047941f1a546dfc42ce997aafe4e/the_desert_triad_remake_v1.0.sd7"

def inspect_header():
    print(f"Downloading {MAP_URL.split('/')[-1]}...")
    r = requests.get(MAP_URL)
    with open("debug_map.sd7", "wb") as f:
        f.write(r.content)
        
    print("Extracting SMF header...")
    smf_found = False
    
    try:
        with py7zr.SevenZipFile("debug_map.sd7", 'r') as z:
            for fname in z.getnames():
                if fname.lower().endswith(".smf"):
                    print(f"Found SMF: {fname}")
                    z.extract(targets=[fname])
                    smf_found = True
                    
                    with open(fname, "rb") as f:
                        # Lees de eerste 64 bytes
                        header_bytes = f.read(64)
                        
                        print("\n--- HEX DUMP (Eerste 64 bytes) ---")
                        # Print in rijen van 16 bytes
                        for i in range(0, 64, 16):
                            chunk = header_bytes[i:i+16]
                            hex_str = " ".join(f"{b:02X}" for b in chunk)
                            print(f"Offset {i:02d}: {hex_str}")

                        print("\n--- INTEGER INTERPRETATION (Little Endian) ---")
                        # Unpack als integers
                        ints = struct.unpack('<16I', header_bytes)
                        labels = [
                            "Magic 1", "Magic 2", "Magic 3", "Magic 4",
                            "Version", "MapID", "MapWidth (??)", "MapHeight (??)",
                            "SquareSize", "Texels", "TileSize", "MinDepth",
                            "MaxDepth", "HeightPtr", "TypePtr", "TileIndexPtr"
                        ]
                        
                        for i, val in enumerate(ints):
                            print(f"Index {i:02d} [{labels[i] if i < len(labels) else '...'}] : {val}")
                            
                    # Opruimen
                    os.remove(fname)
                    break
    except Exception as e:
        print(f"Error: {e}")

    if not smf_found:
        print("Geen .smf bestand gevonden in het archief.")
        
    if os.path.exists("debug_map.sd7"): os.remove("debug_map.sd7")
    if os.path.exists("maps"): shutil.rmtree("maps") # Soms pakt 7z uit in een mapje

if __name__ == "__main__":
    inspect_header()