import sys
from pathlib import Path

MASK_RCNN_LIB = 'Mask_RCNN/'
sys.path.append(MASK_RCNN_LIB)

from src.data_provider import gen_train_val_datasets
from src.config import get_config
import mrcnn.model as modellib
from src.data_provider import seq


TRAIN_SET = Path('data/train/img/')
VAL_SET = Path('data/train/msk')
LOG_DIR = "logging"

def main():
    print('Loading dataset')
    train_set, val_set = gen_train_val_datasets(TRAIN_SET, VAL_SET)
    config = get_config(train_set, val_set)

    print('Creating model')
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=LOG_DIR)

    print('Training model')
    model.train(train_set, val_set,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='all',
                augmentation=seq)



if __name__ == '__main__':
    main()