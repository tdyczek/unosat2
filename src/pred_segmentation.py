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

CLASSIFIER_PREDS = 'preds.csv'
IMAGES_DIR = 'data/test/img/'
OUT_DIR = 'data/test/preds2/'
MODEL_DIR = 'models/1/resnet50_2'


def save_negatives(neg_df, images_dir, out_dir):
    print('Saving false')
    for _, row in tqdm(neg_df.iterrows()):
        im_path = images_dir / row['image']
        out_path = out_dir / row['image']
        im = t.imread(str(im_path))
        out_data = np.zeros(im.shape[:2], dtype='float32')
        t.imsave(str(out_path), out_data)


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

            buildings = preds_np[0] > 0.5
            borders = preds_np[1] > 0.5

            mask = buildings * (~borders)

            shape0 = shape[0].item()
            shape1 = shape[1].item()
            resized_mask = resize(mask, (shape0, shape1), order=0, preserve_range=True).astype(
                'float32'
            )
            all_labels = measure.label(resized_mask, background=0).astype('float32')

            mask_path = out_dir / name[0]
            borders_path = out_dir / f'{name[0]}_borders'
            buildings_path = out_dir / f'{name[0]}_buildings'
            t.imsave(mask_path, all_labels)
            t.imsave(borders_path, borders)
            t.imsave(buildings_path, buildings)


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
