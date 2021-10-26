import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default= '/home/czey/generative_inpainting/training_data/CTArmNpy/validation/', help='path to the dataset')
parser.add_argument('--output', type=str, default='/home/czey/generative_inpainting/data/CTArmNpy/validation_static_view.flist', help='path to the file list')
args = parser.parse_args()

# ext = {'.JPG', '.JPEG', '.PNG', '.TIF', 'TIFF', '.npy'}
ext = {'.npy'}

images = []
for root, dirs, files in os.walk(args.path):
    print('loading ' + root)
    for dir in dirs:
        for file in sorted(os.listdir(os.path.join(root, dir))):
            if os.path.splitext(file)[1].lower() == '.npy':
                images.append(os.path.join(root, dir, file))
# for root, dirs, files in os.walk(args.path):
#     print('loading ' + root)
#     for file in files:
#         if os.path.splitext(file)[1].upper() in ext:
#             images.append(os.path.join(root, file))

images = sorted(images)
np.savetxt(args.output, images, fmt='%s')