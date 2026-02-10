
import os
import glob
import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
INPUT_DIR  = r"F:\Desktop\Minor\Python\Dataset\Espada\Pix2Pix"
OUTPUT_DIR = os.path.join(INPUT_DIR, "ManualROI")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# window name
WIN = "Draw ROI - lclick to add, p=finish, r=reset, q=quit"
# ──────────────────────────────────────────────────────────────────────────────

# globals used by mouse callback
points = []
current_img = None
display_img = None

def mouse_callback(event, x, y, flags, param):
    global points, display_img
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        # draw small circle
        cv2.circle(display_img, (x, y), 3, (0,255,0), -1)
        # draw line to previous pt
        if len(points) > 1:
            cv2.line(display_img, points[-2], points[-1], (0,255,0), 1)
        cv2.imshow(WIN, display_img)

def process_image(path):
    global points, current_img, display_img
    points = []
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[ERROR] could not read {path}")
        return False
    # color copy for drawing
    display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    current_img = img.copy()

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, mouse_callback)

    while True:
        cv2.imshow(WIN, display_img)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('r'):
            # reset
            points = []
            display_img = cv2.cvtColor(current_img, cv2.COLOR_GRAY2BGR)
            cv2.imshow(WIN, display_img)

        elif key == ord('p'):
            # finish polygon if valid
            if len(points) < 3:
                print("Need at least 3 points to form a polygon.")
                continue
            # draw closing line
            cv2.line(display_img, points[-1], points[0], (0,255,0), 1)
            cv2.imshow(WIN, display_img)
            # build mask
            mask = np.zeros_like(current_img, dtype=np.uint8)
            pts_np = np.array(points, np.int32)[None, ...]
            cv2.fillPoly(mask, pts_np, 255)
            # apply mask
            masked = cv2.bitwise_and(current_img, mask)
            # save
            base = os.path.splitext(os.path.basename(path))[0]
            out_mask = os.path.join(OUTPUT_DIR, base + "_roi_mask.png")
            out_img  = os.path.join(OUTPUT_DIR, base + "_roi.png")
            cv2.imwrite(out_mask, mask)
            cv2.imwrite(out_img, masked)
            print(f"Saved: {out_mask}, {out_img}")
            break

        elif key == ord('q'):
            # quit entire script
            cv2.destroyAllWindows()
            return True

    cv2.destroyAllWindows()
    return False

if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.png")))
    for fp in files:
        print("\n===", os.path.basename(fp), "===")
        should_quit = process_image(fp)
        if should_quit:
            print("Quitting early.")
            break
    print("Done. And saved")