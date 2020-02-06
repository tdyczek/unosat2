import sys
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize
from pathlib import Path
from tqdm import tqdm

TO = 572


def main(ims_path):
    for im_path in tqdm(Path(ims_path).glob("*.tif")):
        im = imread(im_path)
        if im.shape[0] != TO or im.shape[1] == TO:
            target_shape = list(im.shape)
            target_shape[0] = TO
            target_shape[1] = TO
            new_im = resize(im, target_shape, order=0, preserve_range=True).astype(
                im.dtype
            )
            assert np.array_equal(np.unique(im), np.unique(new_im))
            imsave(im_path, new_im, check_contrast=False)


if __name__ == "__main__":
    main(sys.argv[1])
