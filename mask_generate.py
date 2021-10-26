import math
import numpy as np
import matplotlib.pyplot as plt

def circlemask_cropped(input_shape):
    # D, H, W, _ = x.shape
    D, H, W, _ = input_shape
    x, y = np.ogrid[:H, :W]
    cx, cy = H / 2, W / 2
    radius = int(np.random.uniform(0.75, 0.75) * H / 2)
    r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
    circmask = r2 > radius * radius
    mask = np.expand_dims(circmask, axis=[0,1,-1]).repeat([D, ], axis=1)
    return mask


if __name__=='__main__':
    mask = circlemask_cropped(512, 512)
    mask2 = mask(512, 512)
    maks_all = (mask + mask2).astype(np.bool)
    plt.imshow(maks_all, cmap='gray')
    plt.show()