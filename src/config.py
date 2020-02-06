from mrcnn.config import Config
from math import ceil


class SatelliteConfig(Config):
    NAME = "satellite_imagery"

    GPU_COUNT = 1
    IMAGE_PER_GPU = 16

    NUM_CLASSES = 1 + 1

    IMAGE_MIN_DIM = 576
    IMAGE_MAX_DIM = 576

    TRAIN_ROIS_PER_IMAGE = 256

    BACKBONE = "resnet50"


def get_config(train_set, val_set):
    config = SatelliteConfig()
    train_steps = ceil(len(train_set.imgs) / config.IMAGE_PER_GPU)
    val_steps = ceil(len(val_set.imgs) / config.IMAGE_PER_GPU)
    config.STEPS_PER_EPOCH = train_steps
    config.VALIDATION_STEPS = val_steps
    return config

class LookupConfig(SatelliteConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
