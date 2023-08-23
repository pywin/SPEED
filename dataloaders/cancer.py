# Originally written by Kazuto Nakashima 
# https://github.com/kazuto1011/deeplab-pytorch

from base import BaseDataSet, BaseDataLoader
from utils import palette
import numpy as np
import os
import scipy
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms

class ZYDataset(BaseDataSet):
    """
    Pascal Voc dataset
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    """
    def __init__(self, **kwargs):
        self.num_classes = 3
        self.palette = palette.get_voc_palette(self.num_classes)
        self.txtRoot = './data'
        super(ZYDataset, self).__init__(**kwargs)

    def _set_files(self):
        # self.root = os.path.join(self.root, 'VOCdevkit/VOC2012')
        self.image_dir = os.path.join(self.root, 'MSI_MSS')
        self.label_dir = os.path.join(self.root, 'MSI_MSS')

        file_list = os.path.join(self.txtRoot, self.split + ".txt")
        self.files = [line.rstrip() for line in tuple(open(file_list, "r"))]
    
    def _load_data(self, index):
        image_id = self.files[index]
        category, numb = image_id.split('_')
        image_path = os.path.join(self.image_dir, category + '_' + 'img' + '_' + numb + '.png')
        label_path = os.path.join(self.label_dir, category + '_' + 'mask' + '_' + numb + '.png')
        image = np.asarray(Image.open(image_path), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        # image_id = self.files[index].split("/")[-1].split(".")[0]
        return image, label, image_id

class ZHEYI(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, val=False,
                    shuffle=False, flip=False, rotate=False, blur=False, augment=False, val_split=None, return_id=False):
        
        self.MEAN = [0.86977233, 0.75382274, 0.87364743]
        self.STD = [0.1357671, 0.24262499, 0.11177172]



        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }

        self.dataset = ZYDataset(**kwargs)
        super(ZHEYI, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)

