import numpy as np
import argparse
import matplotlib.pyplot as plt

from glob import glob
from ntpath import basename
# from scipy.misc import imread
from imageio import imread
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
from skimage.color import rgb2gray
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description='script to compute all statistics')
    parser.add_argument('--data-path', default='/home/czey/generative_inpainting/training_data/CTArmNpy/validation',  help='Path to ground truth data', type=str)
    parser.add_argument('--output-path', default='/home/czey/generative_inpainting/training_data/CTArmNpy/completed_Edge210714', help='Path to output data', type=str)
    # parser.add_argument('--output-path', default='/home/czey/00-pytorch-CycleGAN-xk/test_pix2pix', help='Path to output data', type=str)
    parser.add_argument('--debug', default=0, help='Debug', type=int)
    args = parser.parse_args()
    return args


def compare_mae(img_true, img_test):
    img_true = img_true.astype(np.float32)
    img_test = img_test.astype(np.float32)
    # return np.sum(np.abs(img_true - img_test)) / np.sum(img_true + img_test)
    return np.sum(np.abs(img_true - img_test)) / (img_true.shape[0] * img_true.shape[1])

def compare_mae_mask(img_true, img_test,mask):
    img_true = (img_true*mask).astype(np.float32)
    img_test = (img_test*mask).astype(np.float32)
    # return np.sum(np.abs(img_true - img_test)) / np.sum(img_true + img_test)
    return np.sum(np.abs(img_true - img_test)) / np.sum(mask)


def mean_squared_error_mask(img_true, img_test, mask):
    img_true = (img_true*mask).astype(np.float32)
    img_test = (img_test*mask).astype(np.float32)
    return np.sum((img_true - img_test) ** 2) / np.sum(mask)

def compare_psnr_mask(image_true, image_test, mask, data_range=None):
    err = mean_squared_error_mask(image_true, image_test, mask)
    return 10 * np.log10((data_range ** 2) / err)


args = parse_args()
for arg in vars(args):
    print('[%s] =' % arg, getattr(args, arg))

path_true = args.data_path
path_pred = args.output_path

psnr = []
psnr_mask = []
ssim = []
mae = []
mae_mask = []
names = []
index = 1

# files = list(glob(path_true + '/*.jpg')) + list(glob(path_true + '/*.png'))
files = list(glob(path_true + '/Y180338/*.npy')) + list(glob(path_true + '/Y*/*.mat'))
for fn in sorted(files):
    name = basename(str(fn))
    names.append(name)

    # img_gt = (imread(str(fn)) / 255.0).astype(np.float32)
    img_gt = np.load(fn)
    img_gt[img_gt > 2500] = 2500
    img_gt = img_gt
    # img_pred = (imread(path_pred + '/' + basename(str(fn))) / 255.0).astype(np.float32)
    # img_pred = np.load(path_pred + '/' + basename(str(fn)))
    img_pred = np.load(fn.replace(path_true, path_pred))
    # mask = cv2.imread('/home/czey/00-pytorch-CycleGAN-xk/result_compare/mask/' + basename(str(fn)).replace('npy', 'png'), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread('/home/czey/00-pytorch-CycleGAN-xk/result_compare/mask_0.75.png', cv2.IMREAD_UNCHANGED)
    mask = (mask/255).astype(np.bool)
    img_mask = (mask * img_gt).astype(np.float32)
    mask_small = img_mask > 100
    if sum(sum(mask_small)) ==0:
        continue
    # img_gt = rgb2gray(img_gt)
    # img_pred = rgb2gray(img_pred)

    if args.debug != 0:
        plt.subplot('121')
        plt.imshow(img_gt)
        plt.title('Groud truth')
        plt.subplot('122')
        plt.imshow(img_pred)
        plt.title('Output')
        plt.show()

    psnr.append(compare_psnr(img_gt, img_pred, data_range=2500))
    psnr_mask.append(compare_psnr_mask(img_gt, img_pred, mask_small, data_range=2500))
    ssim.append(compare_ssim(img_gt, img_pred, data_range=2500, win_size=51))
    mae.append(compare_mae(img_gt, img_pred))
    mae_mask.append(compare_mae_mask(img_gt, img_pred, mask_small))
    if np.mod(index, 100) == 0:
        print(
            str(index) + ' images processed',
            "PSNR: %.4f" % round(np.mean(psnr), 4),
            "PSNR_mask: %.4f" % round(np.mean(psnr_mask), 4),
            "PSNR_mask Variance: %.4f" % round(np.var(psnr_mask), 4),
            "SSIM: %.4f" % round(np.mean(ssim), 4),
            "MAE: %.4f" % round(np.mean(mae), 4),
            "MAE_mask: %.4f" % round(np.mean(mae_mask), 4),
            "MAE_mask Variance: %.4f" % round(np.var(mae_mask), 4)
        )
    index += 1

# np.savez(args.output_path + '/metrics.npz', psnr=psnr, ssim=ssim, mae=mae, names=names)
# np.savetxt('/home/czey/00-pytorch-CycleGAN-xk/result_compare/result_PSNR.txt', psnr, fmt='%0.4f')
print(
    "PSNR: %.4f" % round(np.mean(psnr), 4),
    "PSNR Variance: %.4f" % round(np.var(psnr), 4),
    "PSNR_mask: %.4f" % round(np.mean(psnr_mask), 4),
    "PSNR_mask Variance: %.4f" % round(np.var(psnr_mask), 4),
    "SSIM: %.4f" % round(np.mean(ssim), 4),
    "SSIM Variance: %.4f" % round(np.var(ssim), 4),
    "MAE: %.4f" % round(np.mean(mae), 4),
    "MAE Variance: %.4f" % round(np.var(mae), 4),
    "MAE_mask: %.4f" % round(np.mean(mae_mask), 4),
    "MAE_mask Variance: %.4f" % round(np.var(mae_mask), 4)
)
