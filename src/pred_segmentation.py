import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
import tifffile as t
from src.data_provider import SegmentationPredictor
from torch.utils.data import DataLoader
from src.models import UNetResNet18
import torch
from skimage.transform import resize
from skimage import measure
import scipy.ndimage as ndimage
from skimage.morphology import watershed


CLASSIFIER_PREDS = 'preds.csv'
IMAGES_DIR = 'data/test/img/'
OUT_DIR = 'data/test/preds3/'
MODEL_DIR = 'models/1/best_model1'


def save_negatives(neg_df, images_dir, out_dir):
    print('Saving false')
    for _, row in tqdm(neg_df.iterrows()):
        im_path = images_dir / row['image']
        out_path = out_dir / row['image']
        im = t.imread(str(im_path))
        out_data = np.zeros(im.shape[:2], dtype='float32')
        t.imsave(str(out_path), out_data)


def label_watershed(before, after, component_size=20):
    markers = ndimage.label(after)[0]

    labels = watershed(-before, markers, mask=before, connectivity=8)
    unique, counts = np.unique(labels, return_counts=True)

    for (k, v) in dict(zip(unique, counts)).items():
        if v < component_size:
            labels[labels == k] = 0
    return labels

def save_positives(pos_df, images_dir, out_dir, model_dir):
    print('Loading model')
    model = UNetResNet18()
    model.load_state_dict(torch.load(model_dir))
    model.eval().cuda()

    print('Loading dataset')
    dataset = SegmentationPredictor(pos_df['image'].tolist(), images_dir)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)

    with torch.no_grad():
        for name, im, shape in tqdm(dataloader):
            im = im.cuda()
            preds = torch.sigmoid(model(im))

            preds_np = preds.cpu().numpy()[0]

            mask = (preds_np[0] > 0.5).astype(np.uint8)
            contour = preds_np[1]
            seed = ((mask * (1 - contour)) > 0.5).astype(np.uint8)

            shape0 = shape[0].item()
            shape1 = shape[1].item()
            mask = resize(mask, (shape0, shape1), order=0, preserve_range=True).astype(
                'uint8'
            )
            seed = resize(seed, (shape0, shape1), order=0, preserve_range=True).astype(
                'uint8'
            )

            labels = label_watershed(mask, seed)

            mask_path = out_dir / name[0]
            # borders_path = out_dir / f'{name[0]}_borders'
            t.imsave(mask_path, labels)
            # t.imsave(borders_path, contour)


def main(cls_preds_file, images_dir, out_dir, model_dir):
    preds_df = pd.read_csv('preds.csv')
    neg_df = preds_df[preds_df['class'] == 0]
    pos_df = preds_df[preds_df['class'] == 1]
    save_negatives(neg_df, images_dir, out_dir)
    save_positives(pos_df, images_dir, out_dir, model_dir)


if __name__ == '__main__':
    main(cls_preds_file=CLASSIFIER_PREDS,
         images_dir=Path(IMAGES_DIR),
         out_dir=Path(OUT_DIR),
         model_dir=MODEL_DIR)
