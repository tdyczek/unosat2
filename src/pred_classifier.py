import argparse
from pathlib import Path
from src.models import resnet34_classifier
from src.data_provider import ClassifierPredictor
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch_toolbelt.inference import tta
from torch.nn.functional import sigmoid
import argparse


@torch.no_grad()
def main(im_path, model_path, pos_list, neg_list):
    model = resnet34_classifier()
    model.load_state_dict(torch.load(model_path))
    model.eval().cuda()

    pos_paths = []
    neg_paths = []

    dataset = ClassifierPredictor(im_path)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)

    for path, im in tqdm(dataloader):
        im = im.cuda()
        preds = tta.d4_image2label(model, im)
        if sigmoid(preds[0]) > 0.25:
            pos_paths.append(path[0])
        else:
            neg_paths.append(path[0])

    with open(pos_list, "w") as outfile:
        outfile.write("\n".join(pos_paths))

    with open(neg_list, "w") as outfile:
        outfile.write("\n".join(neg_paths))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("im_path", type=int, action="store")
    parser.add_argument("model_path", type=float, action="store")
    parser.add_argument("pos_list", type=str, action="store")
    parser.add_argument("neg_list", type=str, action="store")
    args = parser.parse_args()
    main(args.im_path, args.model_path, args.pos_list, args.neg_list)
