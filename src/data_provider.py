import numpy as np

from imgaug import augmenters as iaa
from skimage.io import imread
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from albumentations import Compose, Normalize, \
    RandomRotate90, VerticalFlip, HorizontalFlip, \
    RandomBrightnessContrast, IAAAdditiveGaussianNoise, \
    CLAHE, RGBShift, RandomGamma, Resize
from torch.utils.data import Dataset
from skimage.transform import resize

test_aug = Normalize()

train_aug = Compose([
    RandomRotate90(),
    VerticalFlip(),
    HorizontalFlip(),
    RGBShift(p=.2),
    RandomGamma(p=.2),
    CLAHE(p=.2),
    RandomBrightnessContrast(p=.2),
    IAAAdditiveGaussianNoise(p=.2),
    Normalize()
])

test_seg_aug = Compose([
    Resize(576, 576, 0, p=1),
    Normalize()
])

train_seg_aug = Compose([
    RandomRotate90(),
    VerticalFlip(),
    HorizontalFlip(),
    RGBShift(p=.2),
    RandomGamma(p=.2),
    CLAHE(p=.2),
    RandomBrightnessContrast(p=.2),
    IAAAdditiveGaussianNoise(p=.2),
    Resize(576, 576, 0),
    Normalize()
])


class ClassifierDataset(Dataset):
    def __init__(self, ims_path: str, msks_path: str, is_train: bool, cls=None):
        self.cls = cls
        self.ims_path = ims_path
        self.msks_path = msks_path
        if is_train:
            self.aug = train_aug
        else:
            self.aug = test_aug
        self.ims = [im_path for im_path in self.ims_path.glob('*.tif')]

    def __len__(self):
        return len(self.ims)

    def __getitem__(self, idx):
        im_path = self.ims[idx]
        raw_img = np.array(imread(im_path))
        aug_img = self.aug(image=raw_img)['image']

        if self.cls == 1:
            cls = 1
        elif self.cls == 0:
            cls = 0
        else:
            msk = imread(self.msks_path / im_path.name)
            cls = np.unique(msk).size > 1

        return aug_img.transpose([2, 0, 1]), float(cls)


class ClassifierPredictor(Dataset):
    def __init__(self, ims_path):
        self.ims_path = ims_path
        self.ims = [im for im in ims_path.glob('*.tif')]
        self.aug = test_aug

    def __len__(self):
        return len(self.ims)

    def __getitem__(self, idx):
        im_path = self.ims[idx]
        raw_img = np.array(imread(im_path))

        aug_img = self.aug(image=raw_img)['image']

        return im_path.name, aug_img.transpose([2, 0, 1])


class SegmentationPredictor(Dataset):
    def __init__(self, names, images_dir):
        self.images_dir = images_dir
        self.names = names
        self.ims_path = [self.images_dir / name for name in names]
        self.aug = test_seg_aug

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        im_path = self.ims_path[idx]
        raw_img = np.array(imread(im_path))

        aug_img = self.aug(image=raw_img)['image']

        return im_path.name, aug_img.transpose([2, 0, 1]), raw_img.shape[:2]


class SegmentationDataset(Dataset):
    def __init__(self, ims_path: str, msks_path: str, is_train: bool, cls=None):
        self.cls = cls
        self.ims_path = ims_path
        self.msks_path = msks_path
        if is_train:
            self.aug = train_seg_aug
        else:
            self.aug = test_seg_aug
        self.ims = [im_path for im_path in self.ims_path.glob('*.tif')]

    def __len__(self):
        return len(self.ims)

    def __getitem__(self, idx):
        im_path = self.ims[idx]
        raw_img = np.array(imread(im_path))
        mask = (np.array(imread(self.msks_path / im_path.name)) > 0).astype(float)

        data = self.aug(image=raw_img, mask=mask)

        aug_img = data['image']
        mask = data['mask']

        return aug_img.transpose([2, 0, 1]), mask[None, ...]
