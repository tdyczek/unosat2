import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_provider import SegmentationDataset
from src.metrics import dice_loss, iou, jaccard


def train(model, optimizer, ims_path, msks_path):
    model = model.train().cuda()
    criterion = nn.BCEWithLogitsLoss()

    losses = []
    ious = []
    jaccs = []

    positives = DataLoader(SegmentationDataset(ims_path / 'true', msks_path, True),
                           batch_size=7,
                           num_workers=3, pin_memory=True, shuffle=True)

    negatives = DataLoader(SegmentationDataset(ims_path / 'false', msks_path, True),
                           batch_size=1,
                           num_workers=1, pin_memory=True, shuffle=True)

    batch_iterator = enumerate(zip(positives, negatives))

    with tqdm(total=len(positives)) as pbar:
        for i, ((x_p, y_p), (x_n, y_n)) in batch_iterator:
            optimizer.zero_grad()

            x = torch.cat((x_p, x_n), 0).cuda()
            y = torch.cat((y_p, y_n), 0).cuda()

            y_pred = model(x)
            y_pred_probs = torch.sigmoid(y_pred).view(-1)
            loss = criterion(y_pred.view(-1), y.view(-1)) + \
                   .1 * dice_loss(y_pred_probs, y)

            iou_ = iou(y_pred_probs.float(), y.byte())
            jacc_ = jaccard(y_pred_probs.float(), y)
            ious.append(iou_.item())
            losses.append(loss.item())
            jaccs.append(jacc_.item())

            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=np.mean(losses),
                             iou=np.mean(ious),
                             jacc=np.mean(jaccs),
                             refresh=True)
            pbar.update()


@torch.no_grad()
def test(model, ims_path, msks_path):
    print(f'Testing')
    model = model.eval().cuda()

    test_set = DataLoader(SegmentationDataset(ims_path, msks_path, False),
                          batch_size=16,
                          num_workers=4, pin_memory=True, shuffle=False)

    losses = []
    ious = []
    jaccs = []
    criterion = nn.BCEWithLogitsLoss()

    for i, (x, y) in enumerate(tqdm(test_set)):
        x, y = x.cuda(), y.cuda()
        y_real = y.view(-1).float()

        y_pred = model(x)
        y_pred_probs = torch.sigmoid(y_pred).view(-1)
        loss = criterion(y_pred.view(-1), y_real) + \
               .1 * dice_loss(y_pred_probs, y_real)

        iou_ = iou(y_pred_probs.float(), y_real.byte())
        jacc_ = jaccard(y_pred_probs.float(), y_real)
        ious.append(iou_.item())
        losses.append(loss.item())
        jaccs.append(jacc_.item())

    print(f'Testing metrics: iou {np.mean(ious)}, '
          f'jacc {np.mean(jaccs)}, '
          f'loss {np.mean(losses)}')
    return 1 - np.mean(ious)
