import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_provider import ClassifierDataset


def train(model, optimizer, ims_path, msks_path):
    print("Training")
    model = model.train().cuda()
    criterion = nn.BCEWithLogitsLoss()

    losses = []
    accuracies = []

    positives = DataLoader(
        ClassifierDataset(ims_path / "true", msks_path, True, 1),
        batch_size=4,
        num_workers=2,
        pin_memory=True,
        shuffle=True,
    )
    negatives = DataLoader(
        ClassifierDataset(ims_path / "false", msks_path, True, 0),
        batch_size=4,
        num_workers=2,
        pin_memory=True,
        shuffle=True,
    )

    batch_iterator = enumerate(zip(positives, negatives))

    with tqdm(total=len(negatives)) as pbar:
        for i, ((x_p, y_p), (x_n, y_n)) in batch_iterator:
            optimizer.zero_grad()

            x = torch.cat((x_p, x_n), 0).cuda()
            y = torch.cat((y_p, y_n), 0).cuda()
            y_pred = model(x).view(-1)
            loss = criterion(y_pred, y)

            losses.append(loss.item())
            accuracies += list(((y_pred > 0) == y.byte()).detach().cpu().numpy())

            loss.backward()
            optimizer.step()

            pbar.set_postfix(
                loss=np.mean(losses), accuracy=np.mean(accuracies), refresh=True
            )
            pbar.update()


@torch.no_grad()
def test(model, ims_path, msks_path):
    print("Testing")
    model = model.eval().cuda()

    test_set = DataLoader(
        ClassifierDataset(ims_path, msks_path, False),
        batch_size=16,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )

    losses = []
    accuracies = []
    criterion = nn.BCEWithLogitsLoss()

    for i, (x, y) in enumerate(tqdm(test_set)):
        x, y = x.cuda(), y.cuda()
        y_pred = model(x)
        loss = criterion(y_pred.view(-1), y.view(-1))

        losses.append(loss.item())
        accuracies += list(
            ((y_pred.view(-1) > 0) == y.view(-1).byte()).detach().cpu().numpy()
        )

    print(
        f"Testing metrics: "
        f"loss {np.mean(losses)} "
        f"accuracy {np.mean(accuracies)}"
    )
    return 1 - np.mean(accuracies)
