from src.data.provider import ImageMaskDataset
from pathlib import Path
import matplotlib.pyplot as plt

TRAIN_IMS = Path("data/train/img/")
TRAIN_MSK = Path("data/train/msk/")


def main():
    dataset = ImageMaskDataset(TRAIN_IMS, TRAIN_MSK)

    for i in range(len(dataset)):
        im, tgt = dataset.__getitem__(i)
        print(i, len(tgt["iscrowd"]))
        plt.imshow(im)
        plt.show()
        plt.imshow(tgt["raw_mask"])
        plt.show()


if __name__ == "__main__":
    main()
