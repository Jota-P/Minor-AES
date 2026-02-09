import os
import glob
import numpy as np
import imageio

# Paths
input_folder = 'F:\Desktop\Minor\Dataset\Videos\Espada\FramesCrop'
output_folder = 'F:\Desktop\Minor\Dataset\Videos\Espada\RF'

os.makedirs(output_folder, exist_ok=True)

for filepath in glob.glob(os.path.join(input_folder, '*.png')):

    img = imageio.imread(filepath).astype(np.float64)
    
    # Beta & Alpha
    beta = img.min()
    sigma2 = img.var()
    alpha = np.sqrt((24.0 / (np.pi**2)) * sigma2)
    
    # Decompression (Paper Diogo)
    y_RF = np.exp((img - beta) / alpha) - 1.0
    
    
    fname = os.path.splitext(os.path.basename(filepath))[0] + '_rf.tiff'
    out_path = os.path.join(output_folder, fname)
    imageio.imwrite(out_path, y_RF.astype(np.float32))
    
print(f"Processed {len(glob.glob(os.path.join(input_folder, '*.png')))} images and saved RF-domain versions to '{output_folder}'.")