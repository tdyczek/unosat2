import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch_toolbelt.losses import JointLoss, WeightedLoss, JaccardLoss
from src.data_provider import SegmentationDataset
from pathlib import Path


def make_loss():
    loss1 = JaccardLoss(mode="multilabel")
    return loss1


def train(model, optimizer, ims_path, msks_path):
    model = model.train().cuda()
    criterion = make_loss()
    jaccard = JaccardLoss(mode="multilabel")

    losses = []
    jaccs = []

    positives = DataLoader(
        SegmentationDataset(ims_path / "true", msks_path, True),
        batch_size=7,
        num_workers=3,
        pin_memory=True,
        shuffle=True,
    )

    with tqdm(total=len(positives)) as pbar:
        for i, (x, y) in enumerate(positives):
            optimizer.zero_grad()

            x = x.cuda()
            y = y.cuda()

            y_pred = model(x)
            loss = criterion(y_pred, y)
            jacc_ = 1 - jaccard(y_pred, y)

            losses.append(loss.item())
            jaccs.append(jacc_.item())

            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=np.mean(losses), jacc=np.mean(jaccs), refresh=True)
            pbar.update()


@torch.no_grad()
def test(model, ims_path, msks_path):
    print(f"Testing")
    model = model.eval().cuda()

    test_set = DataLoader(
        SegmentationDataset(ims_path, msks_path, False),
        batch_size=8,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )

    losses = []
    jaccs = []
    criterion = make_loss()
    jaccard = JaccardLoss(mode="multilabel")

    for i, (x, y) in enumerate(tqdm(test_set)):
        x, y = x.cuda(), y.cuda()
        y_real = y.float()

        y_pred = model(x)

        loss = criterion(y_pred, y_real)

        jacc_ = 1 - jaccard(y_pred, y_real)
        losses.append(loss.item())
        jaccs.append(jacc_.item())

    print(f"Testing metrics: " f"jacc {np.mean(jaccs)}, " f"loss {np.mean(losses)}")
    return 1 - np.mean(jaccs)
