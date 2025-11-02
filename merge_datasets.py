import os
import shutil
from tqdm import tqdm

# Define source and destination directories
src_root = "spectrograms4"
dst_root = "spectrograms"

# List of emotion folders
folders = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]

for folder in folders:
    src_dir = os.path.join(src_root, folder)
    dst_dir = os.path.join(dst_root, folder)
    
    # Ensure destination folder exists
    os.makedirs(dst_dir, exist_ok=True)
    
    # List all files in the source folder
    files = os.listdir(src_dir)
    print(f"\nMoving {len(files)} files from {src_dir} → {dst_dir}")
    
    for file_name in tqdm(files, desc=f"{folder}", unit="file"):
        src_file = os.path.join(src_dir, file_name)
        dst_file = os.path.join(dst_dir, file_name)
        
        # Move the file (overwrite if already exists)
        try:
            shutil.move(src_file, dst_file)
        except Exception as e:
            print(f"❌ Error moving {src_file}: {e}")

print("\n✅ All spectrograms successfully moved to 'spectrograms/'")
