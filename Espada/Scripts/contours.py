import os
import glob

import cv2
import numpy as np
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
INPUT_DIR  = r"F:\Desktop\Minor\Python\Dataset\Espada\Denoised\Pix2Pix"
OUTPUT_DIR = r"F:\Desktop\Minor\Python\Dataset\Espada\Masks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sobel → binary threshold
SOBEL_THRESHOLD = 30   # adjust to your data

# Active contour parameters
SNAKE_PARAMS = {
    "alpha": 0.015,           # elasticity
    "beta": 10,               # rigidity
    "w_line": 0,              # intensity term weight
    "w_edge": 1,              # gradient term weight
    "gamma": 0.001,           # step size
    "max_px_move": 1.0,       # max point displacement per iteration
    "max_num_iter": 2500,     # max iterations
    "convergence": 0.1,       # early stop threshold
    "boundary_condition": "periodic",
}

# Toggle visualization of each step
VISUALIZE = False
# ──────────────────────────────────────────────────────────────────────────────

def get_largest_contour(binary_img):
    """Return the largest external contour in a binary image (or None)."""
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)

def process_frame(filepath):
    """Process one image: rough mask via Sobel+contour, then refine with snake."""
    # 1) Load
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Cannot read image: {filepath}")

    # 2) Sobel gradients
    sx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.hypot(sx, sy)
    grad = np.uint8(grad / grad.max() * 255)

    # 3) Threshold to binary edge map
    _, edges = cv2.threshold(grad, SOBEL_THRESHOLD, 255, cv2.THRESH_BINARY)

    # 4) Find largest contour
    cnt = get_largest_contour(edges)
    if cnt is None:
        print(f"[WARN] No contour found in {os.path.basename(filepath)}")
        return

    # 5) Build rough mask
    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)

    # 6) Optionally refine with active contour
    #    a) Smooth the masked image to reduce noise
    img_smooth = gaussian(masked := cv2.bitwise_and(img, mask), sigma=3)

    #    b) Initialize snake from the rough contour
    snake_init = np.squeeze(cnt).astype(np.float32)
    #    downsample points to ~200 max
    step = max(1, len(snake_init) // 200)
    snake_init = snake_init[::step]

    #    c) Run the snake
    snake = active_contour(
        img_smooth,
        snake_init,
        **SNAKE_PARAMS
    )

    # 7) Build refined mask from snake
    snake_mask = np.zeros_like(img, dtype=np.uint8)
    pts = np.array([snake], np.int32)
    cv2.fillPoly(snake_mask, pts, 255)

    # 8) Combine with original
    final = cv2.bitwise_and(img, snake_mask)

    # 9) Save outputs
    base = os.path.splitext(os.path.basename(filepath))[0]
    cv2.imwrite(os.path.join(OUTPUT_DIR, base + "_mask.png"), mask)
    cv2.imwrite(os.path.join(OUTPUT_DIR, base + "_snake_mask.png"), snake_mask)
    cv2.imwrite(os.path.join(OUTPUT_DIR, base + "_final.png"), final)

    # 10) visualize
    if VISUALIZE:
        fig, axes = plt.subplots(1, 4, figsize=(16,4))
        axes[0].imshow(img,        cmap='gray'); axes[0].set_title('Original')
        axes[1].imshow(edges,      cmap='gray'); axes[1].set_title('Sobel Edges')
        axes[2].imshow(masked,     cmap='gray'); axes[2].set_title('Rough Mask')
        axes[3].imshow(final,      cmap='gray'); axes[3].set_title('Refined Final')
        for ax in axes: ax.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    files = glob.glob(os.path.join(INPUT_DIR, "*.png"))
    if not files:
        print(f"No .png files found in {INPUT_DIR}")
    for f in files:
        process_frame(f)
    print("Done processing all frames.")
