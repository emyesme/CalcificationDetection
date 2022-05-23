# coding: utf8

import random
import time
import struct
import array
import cv2
import numpy as np
import torch
import torch.utils.data as data

class CvROI(data.Dataset):
    """
    OpenCV Region-Of-Interest (ROI) dataset.

    A loader for indexed multi-scale ROIs extracted on-the-fly from a pool of 2D images.

    Either grayscale (8/16 bits) and color images are supported. 
    Images are converted to float w/o rescaling (original values are kept).
    For supported image formats, see OpenCV.

    All images must be contained into one folder.
    Prefixing and suffixing of image filenames are supported.
    Index files must have the following structure (first 2 lines = header):

    roi_width = xx; roi_height = xx; scale_factor = xx
    <image_filename> <px> <py>
    ...

    or

    roi_width = xx; roi_height = xx; scale_factor = xx
    <image_filename> <pyramid layer> <px> <py> <avg intensity>
    ...

    where:
        'roi_width' is the ROI width
        'roi_height' is the ROI height
        'scale_factor' is the scale factor applied to generate successive pyramid layers
        'image_filename' is the image filename (full or partial)
            - the full image path will be 
              img_folder/img_prefix|image_filename|img_suffix
            - a special case (useful for medical datasets) is when the index file has extension '.mcroi'
              thus according to the Nijmegen convention the image file name contains the patient ID in the
              first 8 digits and the exam ID in the next 2 digits, so the full image path will be
              img_folder/image_filename[0:8]/st|image_filename[9:10]/img_prefix|image_filename|img_suffix
        'pyramid layer' is the image pyramid layer starting from 0 (highest res)
            - each low resolution layer is obtained by rescaling the higher layer with 'scale_factor'
        'px' and 'py' are the horizontal and vertical coordinates of the ROI's top-left corner

    Data AUGMENTATION is performed at sampling time so as to have the desired relative class weights.

    Data PREPROCESSING is performed at sampling time. 
    """

    images_pool = {}  # shared pool of all images loaded indexed by name

    def __init__(self, roi_files, img_folder, img_prefix='', img_suffix='', img_channel=None, img_list='', train=False,
                 crossvalid=(1, 1), class_weights=None, class_max_counts=None, class_min_counts=None,
                 preprocessing=None, augmentation=None, verbose=True):
        """
        Args:
            roi_files   (list[string]):     Index files, one for each class.
            img_folder  (string):           Directory with all the images.
            img_prefix  (string):           Prefix to be added to all the images.
            img_suffix  (string):           Suffix to be added to all the images.
            img_channel (int):              Image channel selection according to OpenCV indexing (B = 0, G = 1, R = 2)
            img_list    (string):           Image inclusion list (only load samples if they belong to the images contained in this list)
            train       (bool):             True for training phase.    
            crossvalid  (int, int)          Cross validation (iteration, number of folds), image-based (required 'img_list')
            class_weights (list[float]):    If not empty, class samples will be augmented according to the corresponding relative weights
            class_max_counts (list[int]):   If not empty, determines the max number of samples that can be loaded, for each class
            class_min_counts (list[int]):   If not empty, determines the min number of samples for each class (data augmentation is applied if needed)
            preprocessing (callable):       Preprocessing transform to apply to each sample.
            augmentation (callable):        Augmentation transform to apply to a given sample. If None, data will be simply replicated.
            verbose     (bool):             Enable/disable verbose mode.
        """
        self.rois = []
        self.n_classes = len(roi_files)
        self.class_labels = []
        self.class_counts = np.zeros(len(roi_files), dtype=int)
        self.class_total_counts = np.zeros(len(roi_files), dtype=int)
        self.class_augmentations = []
        self.preprocessing = preprocessing
        self.augmentation = augmentation
        self.verbose = verbose
        self.total_counts = 0
        self.n_channels = 0

        print('\nLoad dataset:')

        # parse image inclusion list if provided
        img_inclusion_list = []
        if img_list is not '':
            with open(img_list) as f:
                for line in f:
                    img_inclusion_list.append(line.rstrip('\r\n'))

        # apply image-based cross-validation if provided
        # if crossvalid[1] > 1:
        #     assert img_inclusion_list, 'Cannot apply image-based cross-validation: image list is empty'
        #     img_folds = partition(img_inclusion_list, crossvalid[1])
        #     prev_len = len(img_inclusion_list)
        #     if train is True:
        #         img_inclusion_list = [item for item in img_inclusion_list if item not in img_folds[crossvalid[0] - 1]]
        #     else:
        #         img_inclusion_list = img_folds[crossvalid[0] - 1]
        #     print(('...%d-cross validation (image-based) is applied' % crossvalid[1]))
        #     print(('...images are %d, but only %d will be loaded' % (prev_len, len(img_inclusion_list))))

        # parse all index files provided
        print(('...load images from %d classes ' % len(roi_files)))
        t0 = time.time()
        for class_i, roi_file in enumerate(roi_files):
            with open(roi_file) as f:

                # first line contains roi size (width, height) and scale factor
                line = next(f)
                tokens = line.rstrip('\r\n').split(';')
                self.roi_w, self.roi_h = int(tokens[0].split('=')[1]), int(tokens[1].split('=')[1])

                # second line contains column titles, skip
                next(f)

                # determine whether this is a medical dataset
                mcroi = roi_file.endswith('.mcroi')

                # parse remaining lines and load the needed images
                for line in f:
                    tokens = line.rstrip('\r\n').split(' ')

                    # skip sample if image is not in the inclusion list
                    if img_inclusion_list and tokens[0] not in img_inclusion_list:
                        continue

                    # add image to the pool if not present
                    if tokens[0] not in self.images_pool:

                        # image path is obtained in a different way for 'mc' (microcalcification) rois (Nijmegen convention) 
                        if mcroi:
                            img_path = img_folder + '/' + tokens[0][0:8] + '/st' + tokens[0][8:10] + '/' + img_prefix + \
                                       tokens[0] + img_suffix
                        else:
                            img_path = img_folder + '/' + tokens[0]

                        # load image (both float32-cvmat and OpenCV formats are supported)
                        if img_path.endswith('.cvmat'):
                            with open(img_path, "rb") as cvmat:
                                img_height = struct.unpack("<i", cvmat.read(4))[0]
                                img_width = struct.unpack("<i", cvmat.read(4))[0]
                                img_channels = struct.unpack("<i", cvmat.read(4))[0]
                                img_len = img_height * img_width * img_channels
                                img_depth = struct.unpack("<i", cvmat.read(4))[0]
                                if img_depth != cv2.CV_32F:
                                    raise ValueError('Unsupported type in .cvmat (only float32 .cvmat are supported)')
                                if img_channels != 1:
                                    raise ValueError(
                                        'Unsupported channels in .cvmat (only 1-channel .cvmat are supported)')
                                a = array.array('f')
                                a.fromfile(cvmat, img_len)
                                img = np.array(a).reshape((img_height, img_width))
                        # load image with unchanged flag (IMPORTANT, since OpenCV applies color/bitdepth conversions by default)
                        else:
                            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

                        if img is None:
                            raise Exception('Cannot load image at ' + img_path)

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

                        # add image to pool
                        self.images_pool[tokens[0]] = img
                        if self.verbose:
                            print(('...image ' + img_path + " loaded and added to pool"))

                    # add current roi to rois list
                    if len(tokens) is 3:
                        px, py = int(tokens[1]), int(tokens[2])
                    elif len(tokens) is 5:
                        px, py = int(tokens[2]), int(tokens[3])
                    else:
                        raise ValueError(
                            'Expected 3 or 5 tokens at line ' + line.rstrip('\r\n') + ' but found ' + str(len(tokens)))
                    self.rois.append(self.images_pool[tokens[0]][py:py + self.roi_h, px:px + self.roi_w])
                    self.class_labels.append(class_i)
                    self.class_counts[class_i] += 1
                    self.total_counts += 1
                    # res = cv2.resize(self.rois[-1], None,fx=10, fy=10, interpolation = cv2.INTER_NEAREST)
                    # cv2.imshow('image', res)
                    # cv2.waitKey(0)

                    # early exit is reached max class count
                    if class_max_counts and class_max_counts[class_i] and self.class_counts[class_i] >= \
                            class_max_counts[class_i]:
                        break
        print(('...TOTAL TIME elapsed  = %.1f seconds' % (time.time() - t0)))

        # calculate class offsets
        self.class_offsets = [0]
        for count in self.class_counts[:-1]:
            self.class_offsets.append(count)

        # calculate class augmentations and total augmentations
        if class_weights and train:
            assert len(class_weights) is len(self.class_counts), \
                str(len(class_weights)) + ' class probabilities have been provided, but there are ' + str(
                    len(self.class_counts)) + ' classes'

            maj_class = self.class_counts.tolist().index(max(self.class_counts))
            for class_i in range(self.n_classes):
                self.class_augmentations.append(int(round(
                    self.class_counts[maj_class] * (class_weights[class_i] / class_weights[maj_class]) -
                    self.class_counts[class_i])))
                assert self.class_augmentations[-1] >= 0, \
                    'Inconsistent class weights: samples can only be added, not removed'
        elif class_min_counts and train:
            assert len(class_min_counts) is len(self.class_counts), \
                str(len(class_min_counts)) + ' class min counts have been provided, but there are ' + str(
                    len(self.class_counts)) + ' classes'

            for class_i in range(self.n_classes):
                self.class_augmentations.append(max(0, class_min_counts[class_i] - self.class_counts[class_i]))

        self.total_augmentations = sum(x for x in self.class_augmentations)

        # finally we know the total amount of data samples (stored and augmented)
        self.total = self.total_augmentations + self.total_counts
        for c in range(self.n_classes):
            self.class_total_counts[c] = self.class_counts[c] + (
                self.class_augmentations[c] if self.class_augmentations else 0)

        # print useful info
        print(('...roi dim (C x W x H) = %d x %d x %d' % (self.n_channels, self.roi_w, self.roi_h)))
        print(('...class counts        = %s' % self.class_counts))
        print(('...class augmentations = %s' % self.class_augmentations))
        print(('...total counts        = %s' % self.total_counts))

        # fit preprocessing, if any
        if self.preprocessing is not None:
            t0 = time.time()
            self.preprocessing.fit(self.rois)
            print(('...preprocessing       = %s, DONE in %.1f seconds' % (self.preprocessing.name(), time.time() - t0)))

        print(('...data augmentation:  = ' + type(augmentation).__name__))

    def __len__(self):
        return self.total

    def __getitem__(self, idx):

        # sample available
        if idx < self.total_counts:
            sample = self.rois[idx]
            label = self.class_labels[idx]
        # sample not available --> data augmentation
        else:
            # get sample class
            idx -= self.total_counts
            label = 0
            for c, augs in enumerate(self.class_augmentations):
                label = c
                if idx < augs:
                    break
                idx -= augs

            # apply augmentation to a randomly extracted sample
            # print self.class_offsets[label] + random.randint(0,self.class_counts[label])
            sample = self.rois[self.class_offsets[label] + random.randint(0, self.class_counts[label] - 1)]
            if self.augmentation is not None:
                sample = self.augmentation(sample)

        # convert to float32 and transform to torch Tensor
        if self.preprocessing is not None:
            sample = torch.from_numpy(np.transpose(self.preprocessing.apply(sample.astype(np.float32)), (2, 0, 1))).float()
        else:
            sample = torch.from_numpy(np.transpose(sample.astype(np.float32), (2, 0, 1))).float()
            
            
        return sample, label




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
