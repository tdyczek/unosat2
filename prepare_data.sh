TRAIN_DIR=data/train/
TRAIN_IMS=data/train/img.zip
TRAIN_MASKS=data/train/msk.zip

unzip $TRAIN_IMS -d $TRAIN_DIR
unzip $TRAIN_MASKS -d $TRAIN_DIR

export PYTHONPATH=$(pwd)
python src/data_utils/resize_imagery.py $TRAIN_DIR/msk
python src/data_utils/resize_imagery.py $TRAIN_DIR/img

