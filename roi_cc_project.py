# coding: utf8

import random
import time
import struct
import array
import cv2
import numpy as np
import torch
import torch.utils.data as data



class UnlabeledImageROI(data.Dataset):
    """
    Region-Of-Interest (ROI) loader from a single unlabeled 2D image.
    """

    def __init__(self, img_path, gt_path, ROI_size, img_channel=None, preprocessing=None, verbose=True):
        """
        Args:
            img_path    (string):       Image file path
            ROI_size  (int, int):       ROI size (x,y)
            img_channel (int):          Image channel selection according to OpenCV indexing (B = 0, G = 1, R = 2)
            preprocessing (callable):   Preprocessing transform to apply to each sample.
            verbose     (bool):         Enable/disable verbose mode.
        """
        self.rois = []
        self.rois_centers = []
        self.class_labels = []
        self.total = 0
        self.preprocessing = preprocessing
        self.verbose = verbose
        self.n_channels = 0

        # load image
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise Exception(f'Cannot load image at {img_path}')

        if mask is None:
            raise Exception(f'Cannot load image at {gt_path}')

        # preallocate probability image
        self.prob_image = np.zeros(img.shape, dtype=np.single)

        # store/check channels
        img_channels = 1 if len(img.shape) == 2 else img.shape[2]
        if self.n_channels:
            if self.n_channels != img_channels:
                raise Exception('Images have different number of channels')
        else:
            self.n_channels = img_channels

        # select image channel if required (using NumPy instead of cv2.split, more efficient)
        if img_channel is not None and len(img.shape) is 3:
            img = img[:, :, img_channel]

        # add channel dimension if missing (e.g. grayscale samples)
        if len(img.shape) is 2:
            img = img[..., np.newaxis]

        # extract ROIs
        t0 = time.time()
        h = img.shape[0]
        w = img.shape[1]
        roi_w = ROI_size[0]
        roi_h = ROI_size[1]
        for y in range(0, h-roi_h):
            for x in range(0, w-roi_w):
                if img[y, x] > 0:
                    if np.sum(mask[y:y + roi_h, x:x + roi_w]) >0:
                        self.rois.append(img[y:y + roi_h, x:x + roi_w])
                        self.rois_centers.append((int(x + roi_w/2), int(y + roi_h/2)))
                        self.total += 1
                        self.class_labels.append(1)
                    else:
                        self.rois.append(img[y:y + roi_h, x:x + roi_w])
                        self.rois_centers.append((int(x + roi_w/2), int(y + roi_h/2)))
                        self.total += 1
                        self.class_labels.append(0)                        
        print(f'...{len(self.rois)} ROIs extracted in {time.time()-t0:.0f} s')

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        # convert to float32 and transform to torch Tensor
        sample = self.rois[idx]
        if self.preprocessing is not None:
            sample = torch.from_numpy(np.transpose(self.preprocessing.apply(sample.astype(np.float32)), (2, 0, 1))).float()
        else:
            sample = torch.from_numpy(np.transpose(sample.astype(np.float32), (2, 0, 1))).float()
        return sample, self.class_labels[idx]

    def store(self, outputs, indexes):
        for (out, idx) in zip(outputs, indexes):
            self.prob_image[self.rois_centers[idx][1], self.rois_centers[idx][0]] = out
