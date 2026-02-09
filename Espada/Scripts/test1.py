import os
import glob
import cv2
import csv

# ──────────────────────────────────────────────────────────────────────────────
# PATHS (modify if needed)
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR    = r"F:\Desktop\Minor\Python\Dataset\Espada\Areas"
MASK_DIR    = os.path.join(BASE_DIR, "Masks")
CROP_DIR    = os.path.join(BASE_DIR, "Cropped")
OUTPUT_DIR  = os.path.join(BASE_DIR, "Output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# IMAGE EXTENSION for matching (change if yours are .jpg, etc.)
IMG_EXT     = ".png"
# ──────────────────────────────────────────────────────────────────────────────

def process_one(mask_path):
    # derive base filename
    base = os.path.splitext(os.path.basename(mask_path))[0]

    # load mask & binarize
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"[WARN] could not read mask {mask_path}")
        return None
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # compute area (# of nonzero pixels)
    area_px = int(cv2.countNonZero(mask_bin))

    # load corresponding full image
    img_path = os.path.join(CROP_DIR, base + IMG_EXT)
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] no matching image for {base} in Cropped, skipping crop")
        crop = None
    else:
        # find contours to compute bounding box
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"[WARN] no contours found in mask {base}")
            crop = None
        else:
            # pick largest contour and its bounding rect
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            crop = img[y:y+h, x:x+w]
            # save the crop
            out_path = os.path.join(OUTPUT_DIR, base + "_crop" + IMG_EXT)
            cv2.imwrite(out_path, crop)

    return base, area_px

def main():
    masks = sorted(glob.glob(os.path.join(MASK_DIR, "*" + IMG_EXT)))
    if not masks:
        print("No masks found in", MASK_DIR)
        return

    # open CSV writer
    csv_path = os.path.join(OUTPUT_DIR, "areas.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "area_pixels"])
        for m in masks:
            result = process_one(m)
            if result is not None:
                writer.writerow(result)

    print("Done!")
    print("  – Crops & any warnings above")
    print("  – Areas CSV at:", csv_path)

if __name__ == "__main__":
    main()