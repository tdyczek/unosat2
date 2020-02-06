import numpy as np

from imgaug import augmenters as iaa
from skimage.io import imread
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
from mrcnn import utils

sometimes = lambda aug: iaa.Sometimes(0.2, aug)

seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Rot90([0, 1, 2, 3], keep_size=False),
    sometimes(iaa.GammaContrast((0.5, 2.0))),
    sometimes(iaa.AllChannelsCLAHE()),
])


class SatelliteDataset(utils.Dataset):
    def load_data(self, ims_list, mask_path):
        self.add_class("satellite_imagery", 1, "building")
        self.imgs = ims_list
        self.masks = [mask_path / f.name for f in self.imgs]
        for i, im_path in enumerate(self.imgs):
            self.add_image('satellite_imagery', image_id=i, path=im_path)

    def load_image(self, idx):
        img = imread(self.imgs[idx])
        return img

    def load_mask(self, idx):
        mask = imread(self.masks[idx])
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        masks = (mask == obj_ids[:, None, None]).transpose([1, 2, 0])
        class_ids = np.ones(masks.shape[2])
        return masks.astype(bool), class_ids.astype(np.int32)


def gen_train_val_datasets(im_path, mask_path):
    all_ims = []
    all_vals = []
    for f in tqdm(im_path.glob('*.tif')):
        uni_count = np.unique(imread(f)).size
        all_vals.append(uni_count)
        all_ims.append(f)
    all_vals = np.array(all_vals).reshape(-1, 1)
    all_vals = KBinsDiscretizer(encode='ordinal').fit_transform(all_vals)
    paths_train, paths_test = train_test_split(all_ims, random_state=0, train_size=.85, stratify=all_vals)
    train_data = SatelliteDataset()
    test_data = SatelliteDataset()
    train_data.load_data(paths_train, mask_path)
    test_data.load_data(paths_test, mask_path)
    train_data.prepare()
    test_data.prepare()
    return train_data, test_data
