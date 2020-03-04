import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
from src.train_classifier import train as train_classifier, test as test_classifier
from src.train_segmentation import test as test_segmentation, train as train_segmentation
from src.models import resnet18_classifier, UNetResNet18

OUT_PATH = 'models/1/'
TRAIN_PATH = Path('data/train/img/')
VAL_PATH = Path('data/train/img_val/')
MASK_PATH = Path('data/train/msk/')


def save_model(model, epoch, out_path):
    model_name = f"resnet50_{epoch}"
    path = Path(out_path) / model_name
    torch.save(model.state_dict(), path)


def train_test_loop(task, epochs, init_lr, out_path, train_path, val_path):
    print('Loading model')
    if task == 'classification':
        model = resnet18_classifier()
        model.load_state_dict(torch.load('models/resnet18_0.957/resnet18_15'))
        test = test_classifier
        train = train_classifier
    elif task == 'segmentation':
        model = UNetResNet18()
        model.load_state_dict(torch.load('models/1/resnet50_20'))
        model.freeze()
        test = test_segmentation
        train = train_segmentation

    else:
        raise Exception()

    test(model, VAL_PATH, MASK_PATH)

    optimizer = Adam(model.parameters(), lr=init_lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    min_loss = sys.maxsize

    for epoch in range(21, epochs):
        print(f'Epoch {epoch}, lr {optimizer.param_groups[0]["lr"]}')

        if epoch > 20 and task == 'segmentation':
            model.unfreeze()


        train(model, optimizer, TRAIN_PATH, MASK_PATH)
        test_loss = test(model, VAL_PATH, MASK_PATH)
        scheduler.step(test_loss)
        if test_loss < min_loss:
            min_loss = test_loss
            save_model(model, epoch, out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('epochs', type=int, action='store')
    parser.add_argument('init_lr', type=float, action='store')
    parser.add_argument('task', type=str, action='store')
    args = parser.parse_args()

    train_test_loop(args.task, args.epochs, args.init_lr, OUT_PATH, TRAIN_PATH, VAL_PATH)
