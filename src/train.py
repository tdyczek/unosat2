import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
from src.train_classifier import train as train_classifier, test as test_classifier
from src.train_segmentation import (
    test as test_segmentation,
    train as train_segmentation,
)
from src.unet_models import AlbuNet, resnet34_classifier


def save_model(model, epoch, out_path):
    model_name = f"albunet_{epoch}"
    path = Path(out_path) / model_name
    torch.save(model.state_dict(), path)


def train_test_loop(
    task, epochs, init_lr, out_path, train_path, val_path, mask_path, classifier_path
):
    print("Loading model")
    if task == "classification":
        model = resnet34_classifier()
        test = test_classifier
        train = train_classifier
    elif task == "segmentation":
        class_model = resnet34_classifier()
        class_model.load_state_dict(torch.load(classifier_path))
        model = AlbuNet(class_model, num_classes=2, pretrained=True)
        test = test_segmentation
        train = train_segmentation
    else:
        raise Exception()

    test(model, val_path, mask_path)

    optimizer = Adam(model.parameters(), lr=init_lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.2)

    min_loss = sys.maxsize

    for epoch in range(epochs):
        print(f'Epoch {epoch}, lr {optimizer.param_groups[0]["lr"]}')

        train(model, optimizer, train_path, mask_path)
        test_loss = test(model, val_path, mask_path)
        scheduler.step(test_loss)
        if test_loss < min_loss:
            min_loss = test_loss
            save_model(model, epoch, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("epochs", type=int, action="store")
    parser.add_argument("init_lr", type=float, action="store")
    parser.add_argument("task", type=str, action="store")
    parser.add_argument("out_path", type=str, action="store")
    parser.add_argument("train_path", type=str, action="store")
    parser.add_argument("val_path", type=str, action="store")
    parser.add_argument("mask_path", type=str, action="store")
    parser.add_argument("classifier_path", type=str, action="store")
    args = parser.parse_args()

    train_test_loop(
        args.task,
        args.epochs,
        args.init_lr,
        args.out_path,
        args.train_path,
        args.val_path,
        args.mask_path,
        args.classifier_path,
    )
