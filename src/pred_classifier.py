import argparse
from pathlib import Path
from src.models import resnet18_classifier
from src.data_provider import ClassifierPredictor
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch_toolbelt.inference import tta
from torch.nn.functional import sigmoid

IM_PATH = Path('data/test/img/')
MODEL_PATH = 'models/resnet18_0.96/resnet18_best'
POS_LIST = 'logging/pos.txt'
NEG_LIST = 'logging/neg.txt'


@torch.no_grad()
def main(im_path, model_path, pos_list, neg_list):
    model = resnet18_classifier()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval().cuda()

    pos_paths = []
    neg_paths = []

    dataset = ClassifierPredictor(IM_PATH)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)

    for path, im in tqdm(dataloader):
        im = im.cuda()
        preds =  tta.d4_image2label(model, im)
        if sigmoid(preds[0]) > 0.25:
            pos_paths.append(path[0])
        else:
            neg_paths.append(path[0])

    with open(pos_list, "w") as outfile:
        outfile.write("\n".join(pos_paths))

    with open(neg_list, "w") as outfile:
        outfile.write("\n".join(neg_paths))


if __name__ == '__main__':
    main(IM_PATH, MODEL_PATH, POS_LIST, NEG_LIST)
