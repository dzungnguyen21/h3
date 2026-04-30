import subprocess, os

# Install required tools
subprocess.run(["pip", "install", "pycocotools", "--break-system-packages", "-q"])

# Create data directories
os.makedirs("coco/annotations", exist_ok=True)
os.makedirs("coco/images/val2014", exist_ok=True)

# Download COCO 2014 validation annotations (contains object labels)
subprocess.run([
    "wget", "-q", "-O", "coco/annotations/instances_val2014.json",
    "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
])
# Or if wget is available:
os.system("cd coco/annotations && wget -q http://images.cocodataset.org/annotations/annotations_trainval2014.zip && unzip -q annotations_trainval2014.zip")


import requests
from PIL import Image
from io import BytesIO
import json
from tqdm.auto import tqdm
from pycocotools.coco import COCO


# Load COCO API
coco = COCO("coco/annotations/annotations/instances_val2014.json")

# Get 500 image ids
max_images=500
img_ids = coco.getImgIds()[:max_images]
imgs    = coco.loadImgs(img_ids)

# Download images
print("Downloading COCO val2014 images...")
for img_info in tqdm(imgs):
    save_path = f"coco/images/val2014/{img_info['file_name']}"
    if os.path.exists(save_path):
        continue
    url      = img_info["coco_url"]
    response = requests.get(url, timeout=10)
    with open(save_path, "wb") as f:
        f.write(response.content)

print(f"Downloaded {len(imgs)} images.")