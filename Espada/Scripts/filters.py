import cv2
import os
import numpy as np
from skimage.restoration import (
    denoise_wavelet,
    denoise_nl_means,
    estimate_sigma
)
from bm3d import bm3d, BM3DProfile

# — make sure you’ve installed:
#    pip install scikit-image PyWavelets

# Paths
#ADD your paths
input_dir  = r"F:\Desktop\Minor\Python\Dataset\Espada\FramesCrop"
output_base = r"F:\Desktop\Minor\Python\Dataset\Espada\Denoised"

# 1) Mean filter
def apply_mean(img, ksize=5):
    return cv2.blur(img, (ksize, ksize))

# 2) Median filter
def apply_median(img, ksize=5):
    return cv2.medianBlur(img, ksize)

# 3) Gaussian
def apply_gaussian(img, ksize=5, sigma=1.0):
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)

# 4) Bilateral
def apply_bilateral(img, d=9, sigmaColor=75, sigmaSpace=75):
    return cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)

# 5) Non-Local Means (skimage)
def apply_nl_means(img, h=None, patch_size=5, patch_distance=6, fast_mode=True):
    img_f = img.astype(np.float32) / 255.0
    # estimate noise sigma on a single‐channel image
    sigma_est = np.mean(estimate_sigma(img_f, channel_axis=None))
    if h is None:
        h = 1.15 * sigma_est
    den = denoise_nl_means(
        img_f,
        h=h,
        patch_size=patch_size,
        patch_distance=patch_distance,
        channel_axis=None,
        fast_mode=fast_mode
    )
    return (den * 255).astype(np.uint8)

# 6) Wavelet denoise (skimage + PyWavelets)
def apply_wavelet(img, method='BayesShrink', mode='soft'):
    img_f = img.astype(np.float32) / 255.0
    den = denoise_wavelet(
        img_f,
        method=method,
        mode=mode,
        rescale_sigma=True
        # no need to specify channel_axis for single channel
    )
    return (den * 255).astype(np.uint8)

def apply_bm3d(img, sigma_psd=None):
    # convert to [0,1]
    img_f = img.astype(np.float32) / 255.0
    # estimate noise level if needed
    if sigma_psd is None:
        sigma_psd = np.mean(estimate_sigma(img_f, channel_axis=None))
    # run both stages by default
    den = bm3d(img_f, sigma_psd=sigma_psd)
    # back to [0,255] uint8
    return np.clip(den * 255, 0, 255).astype(np.uint8)

# add it to your filters map
filters = {
    'mean':      apply_mean,
    'median':    apply_median,
    'gaussian':  apply_gaussian,
    'bilateral': apply_bilateral,
    'nl_means':  apply_nl_means,
    'wavelet':   apply_wavelet,
    'bm3d':      apply_bm3d
}

# create the BM3D folder
os.makedirs(os.path.join(output_base, 'bm3d'), exist_ok=True)

# … (rest of your loop) …
for fn in os.listdir(input_dir):
    if not fn.lower().endswith(('.png','.jpg','.jpeg','.tif','.bmp')):
        continue
    img = cv2.imread(os.path.join(input_dir, fn), cv2.IMREAD_GRAYSCALE)
    for name, func in filters.items():
        try:
            out = func(img)
            cv2.imwrite(os.path.join(output_base, name, fn), out)
        except Exception as e:
            print(f"[{name}] on {fn}: {e}")
