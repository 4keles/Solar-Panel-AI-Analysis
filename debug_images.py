import cv2
from pathlib import Path

def main():
    p = Path("data/raw_data/captured")
    if not p.exists():
        print("Folder does not exist")
        return
    imgs = list(p.rglob("*.tiff")) + list(p.rglob("*.tif")) + list(p.rglob("*.jpg")) + list(p.rglob("*.png"))
    if not imgs:
        print("No images found")
        return
    
    img_path = str(imgs[0])
    print(f"Reading: {img_path}")
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("cv2.imread returned None!")
    else:
        print(f"Shape: {img.shape}, dtype: {img.dtype}")

if __name__ == "__main__":
    main()
