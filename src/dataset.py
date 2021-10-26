import os
import glob
import scipy
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from imageio import imread
from .utils import circlemask_cropped


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, flist, edge_flist, mask_flist, augment=True, training=True):
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.data = self.load_flist(flist)
        self.edge_data = self.load_flist(edge_flist)
        self.input_size = config.INPUT_SIZE
        self.edge = config.EDGE

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            print('index:', index)
            item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):
        img = np.load(self.data[index])
        img[img > 3000] = 3000
        img = img / 3000
        mask = self.load_mask()
        edge = self.load_edge(index) / 255
        return self.to_tensor(img),  self.to_tensor(edge), self.to_tensor(mask)

    def load_edge(self, index):
        edge = imread(self.edge_data[index])
        return edge

    def load_mask(self):
        mask1 = circlemask_cropped((512, 512))
        mask = mask1.astype(np.bool)
        mask = (mask > 0).astype(np.float)
        return mask

    def to_tensor(self, img):
        img_t = F.to_tensor(img).float()
        return img_t

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist
        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.npy'))
                flist.sort()
                return flist
            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]
        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item
