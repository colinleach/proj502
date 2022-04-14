import os
import logging
import glob
from typing import List
from pathlib import Path

import pandas as pd

import tensorflow as tf

from zoobot.shared import label_metadata
from zoobot.tensorflow.data_utils import image_datasets
from zoobot.tensorflow.estimators import define_model, preprocess
from zoobot.tensorflow.predictions import predict_on_tfrecords, predict_on_dataset
from data.utils import read_params


class MakePredictions:
    """

    """

    def __init__(self, params: Path, file_format: str = 'png', initial_size: int = 300):
        self.file_format = file_format
        self.initial_size = initial_size
        self.crop_size = None
        self.resize_size = None
        self.channels = None

        logfile = Path(params['dataroot']) / 'logfiles/predictions.log'
        logging.basicConfig(
            filename=logfile,
            format='%(asctime)s %(levelname)s: %(message)s',
            level=logging.INFO
        )
        logging.info("----------------------------")

        # useful to avoid errors on small GPU
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
              tf.config.experimental.set_memory_growth(gpu, True)

    def images_from_dir(self, folder: str):
        """
        List the images to make predictions on.
        """

        # utility function to easily list the images in a folder.
        unordered_image_paths: List = predict_on_dataset.paths_in_folder(folder,
                                                                   file_format=self.file_format,
                                                                   recursive=False)
        self.paths_to_dataset(unordered_image_paths)

    def images_from_catalog(self, df: pd.DataFrame):
        ## or maybe you already have a list from a catalog?
        unordered_image_paths = df['file_loc']
        self.paths_to_dataset(unordered_image_paths)

    def paths_to_dataset(self,
                         unordered_image_paths,
                         initial_size: int = 300,
                         batch_size = 32):
        """
        Load the images as a tf.dataset, just as for training

        :param unordered_image_paths:
        :param initial_size: 300 for paper, from tfrecord or from png
                (png will be resized when loaded, before preprocessing)
        :param batch_size: 128 for paper, you'll need a very good GPU.
                8 for debugging, 64 for RTX 2070, 256 for A100
        :return:
        """

        assert len(unordered_image_paths) > 0
        assert os.path.isfile(unordered_image_paths[0])

        raw_image_ds = image_datasets.get_image_dataset([str(x) for x in unordered_image_paths],
                                                        self.file_format,
                                                        initial_size,
                                                        batch_size)

        preprocessing_config = preprocess.PreprocessingConfig(
            label_cols=[],  # no labels are needed, we're only doing predictions
            input_size=initial_size,
            make_greyscale=True,
            normalise_from_uint8=True   # False for tfrecords with 0-1 floats,
                                        # True for png/jpg with 0-255 uints
            )
        self.image_ds = preprocess.preprocess_dataset(raw_image_ds, preprocessing_config)
        # image_ds will give batches of (images, paths) when label_cols=[]

    def set_existing_dataset(self, image_ds, initial_size: int = 300):
        self.image_ds = image_ds
        self.initial_size = initial_size

    def model_params(self, resize_size: int = 224, channels: int = 3):
        """
        Define the model and load the weights.
        You must define the model exactly the same way as when you trained it.
        """
        self.crop_size = int(self.initial_size * 0.75)
        self.resize_size = resize_size  # 224 for paper
        self.channels = channels

    def predict_pretrained(self, checkpoint_loc, save_loc):
        """
        If you're just using the full pretrained Galaxy Zoo model, without finetuning,
        you can just use include_top=True.
        """

        model = define_model.load_model(
            checkpoint_loc=checkpoint_loc,
            include_top=True,
            input_size=self.initial_size,
            crop_size=self.crop_size,
            resize_size=self.resize_size,
            expect_partial=True  # optimiser state will not load as we're not using it for predictions
        )

        label_cols = label_metadata.decals_label_cols

        # save_loc = 'data/results/make_predictions_example.hdf5'
        n_samples = 5
        predict_on_dataset.predict(self.image_ds, model, n_samples, label_cols, save_loc)

    def predict_finetuned(self):
        """
        If you have done finetuning, use include_top=False and replace the output layers exactly as you did when training.
        For example, below is how to load the model in finetune_minimal.py.
        """

        finetuned_dir = 'results/finetune_advanced/full/checkpoint'
        base_model = define_model.load_model(
          finetuned_dir,
          include_top=False,
          input_size=self.initial_size,
          crop_size=self.crop_size,
          resize_size=self.resize_size,
          output_dim=None
        )
        new_head = tf.keras.Sequential([
          tf.keras.layers.InputLayer(input_shape=(7,7,1280)),
          tf.keras.layers.GlobalAveragePooling2D(),
          tf.keras.layers.Dropout(0.75),
          tf.keras.layers.Dense(64, activation='relu'),
          tf.keras.layers.Dropout(0.75),
          tf.keras.layers.Dense(64, activation='relu'),
          tf.keras.layers.Dropout(0.75),
          tf.keras.layers.Dense(1, activation="sigmoid", name='sigmoid_output')
        ])
        model = tf.keras.Sequential([
          tf.keras.layers.InputLayer(input_shape=(self.initial_size, self.initial_size, 1)),
          base_model,
          new_head
        ])
        define_model.load_weights(model, finetuned_dir, expect_partial=True)

        label_cols = ['ring']

        # save_loc = 'data/results/make_predictions_example.csv'
        save_loc = 'data/results/make_predictions_example.hdf5'
        n_samples = 5
        predict_on_dataset.predict(self.image_ds, model, n_samples, label_cols, save_loc)


def gz2_catalog():
    params = read_params()
    mp = MakePredictions(params)
    dataroot = Path(params['dataroot'])
    datadir = dataroot / 'shards/gz2/test_shards'
    df = pd.read_csv(datadir / 'test_df.csv')
    id_loc = df[['id_str', 'file_loc']].copy()
    checkpoint_loc = dataroot / 'results/best_training/gz2/checkpoint'
    save_loc = dataroot / 'results/predictions/gz2.hdf5'
    mp.images_from_catalog(id_loc)
    mp.model_params()
    mp.predict_pretrained(checkpoint_loc, str(save_loc))

def decals_catalog():
    params = read_params()
    mp = MakePredictions(params)
    dataroot = Path(params['dataroot'])
    datadir = dataroot / 'shards/decals/test_shards'
    df = pd.read_csv(datadir / 'test_df.csv')
    id_loc = df[['id_str', 'file_loc']].copy()
    checkpoint_loc = dataroot / 'results/best_training/decals/checkpoint'
    save_loc = dataroot / 'results/predictions/decals.hdf5'
    mp.images_from_catalog(id_loc)
    mp.model_params()
    mp.predict_pretrained(checkpoint_loc, str(save_loc))

# def decals_shards():
#     mp = MakePredictions()
#     params = read_params()
#     dataroot = Path(params['dataroot'])
#     datadir = dataroot / 'shards/decals/test_shards'
#     shards = list(datadir.glob('*.tfrecord'))
#     mp.set_existing_dataset(tf.data.TFRecordDataset(shards[0]))
#     mp.model_params()
#     checkpoint_loc = dataroot / 'results/decals/checkpoint'
#     save_loc = dataroot / 'results/make_predictions_example.hdf5'
#     mp.predict_pretrained(checkpoint_loc, save_loc)

if __name__ == '__main__':
    gz2_catalog()
