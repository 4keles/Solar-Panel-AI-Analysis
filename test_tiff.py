import cv2
from pathlib import Path

def main():
    p = Path("data")
    tiffs = list(p.rglob("*.tiff")) + list(p.rglob("*.tif"))
    if not tiffs:
        print("No TIFFs found in data folder!")
        return
        
    for f in tiffs[:5]:
        print(f"Testing {f}:")
        img_unchanged = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        img_any = cv2.imread(str(f), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        img_color = cv2.imread(str(f), cv2.IMREAD_COLOR)
        
        print(f"  UNCHANGED: {img_unchanged.shape if img_unchanged is not None else 'None'}")
        print(f"  ANYDEPTH:  {img_any.shape if img_any is not None else 'None'}")
        print(f"  COLOR:     {img_color.shape if img_color is not None else 'None'}")

if __name__ == "__main__":
    main()
