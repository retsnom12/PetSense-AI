import os
import shutil
import glob

SOURCE_DIR = "raw_catmeows"  # İndirdiğin wav dosyalarının olduğu klasör

MAPPING = {
    "F_": "data/hungry",
    "I_": "data/stressed",
    "B_": "data/relaxed"
}

for prefix, target_dir in MAPPING.items():
    os.makedirs(target_dir, exist_ok=True)
    files = glob.glob(os.path.join(SOURCE_DIR, f"{prefix}*.wav"))
    for f in files:
        shutil.copy(f, target_dir)
    print(f"Copied {len(files)} files to {target_dir}")