  
import os
import logging
import contextlib
from time import time
from datetime import timedelta, datetime
from pathlib import Path

import tensorflow as tf

from zoobot.tensorflow.data_utils import tfrecord_datasets
from zoobot.tensorflow.training import training_config, losses
from zoobot.tensorflow.estimators import preprocess, define_model
from zoobot.shared import schemas, label_metadata

from projcode.data.utils import read_params

class TrainModel:

    def __init__(self):

        logging.basicConfig(
            filename='training.log',
            format='%(asctime)s %(levelname)s:%(message)s',
            level=logging.INFO
        )
        logging.info("----------------------------")
        self.physical_devices()

    def physical_devices(self):

        # useful to avoid errors on small GPU
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
              tf.config.experimental.set_memory_growth(gpu, True)

        # check which GPU we're using
        physical_devices = tf.config.list_physical_devices('GPU')
        logging.info('GPUs: {}'.format(physical_devices))

    def set_paths(self,
                  shards_dir: Path,
                  save_dir: str):
        self.train_records_dir = shards_dir / 'train_shards'
        self.val_records_dir = shards_dir / 'val_shards'
        self.save_dir = save_dir

        assert save_dir is not None
        if not os.path.isdir(save_dir):
          os.mkdir(save_dir)

    def set_schema(self, pair_type: str):
        pair_types = ['gz2', 'gz2_partial', 'decals', 'decals_partial']
        if pair_type == 'gz2':
            question_answer_pairs = label_metadata.gz2_pairs
        elif pair_type == 'gz2_partial':
            question_answer_pairs = label_metadata.gz2_partial_pairs
        elif pair_type == 'decals':
            question_answer_pairs = label_metadata.decals_pairs
        elif pair_type == 'decals_partial':
            question_answer_pairs = label_metadata.decals_partial_pairs
        else:
            raise ValueError(f"shard_type {pair_type}, invalid, must be in {pair_type}")
        logging.info(f'Using {pair_type} Q&A pairs')

        dependencies = label_metadata.get_gz2_and_decals_dependencies(question_answer_pairs)
        self.schema = schemas.Schema(question_answer_pairs, dependencies)
        logging.info('Schema: {}'.format(self.schema))

    def set_channels(self, color: bool = False):
        # a bit awkward, but I think it is better to have to specify you def. want color than that you def want greyscale
        self.greyscale = not color
        if self.greyscale:
          logging.info('Converting images to greyscale before training')
          self.channels = 1
        else:
          logging.warning('Training on color images, not converting to greyscale')
          self.channels = 3

    def set_context_manager(self, distributed: bool = False):
        if distributed:
            logging.info('Using distributed mirrored strategy')
            strategy = tf.distribute.MirroredStrategy()  # one machine, one or more GPUs
            # strategy = tf.distribute.MultiWorkerMirroredStrategy()  # one or more machines.
            # Not tested - you'll need to set this up for your own cluster.
            self.context_manager = strategy.scope()
            logging.info('Replicas: {}'.format(strategy.num_replicas_in_sync))
        else:
            logging.info('Using single GPU, not distributed')
            self.context_manager = contextlib.nullcontext()  # does nothing, just a convenience for clean code

    def train(self,
             initial_size: int,
             resize_size: int,
             batch_size: int,
             epochs: int,
             dropout_rate: float = 0.2,
             always_augment: bool = False,
             eager: bool = False
             ):

        start = time()
        logging.info(f"Started training with epochs={epochs}, batch_size={batch_size}")

        train_records = [os.path.join(self.train_records_dir, x)
                         for x in os.listdir(self.train_records_dir) if x.endswith('.tfrecord')]
        val_records = [os.path.join(self.val_records_dir, x)
                        for x in os.listdir(self.val_records_dir) if x.endswith('.tfrecord')]

        raw_train_dataset = tfrecord_datasets.get_tfrecord_dataset(train_records,
                                                                   self.schema.label_cols, batch_size,
                                                                   shuffle=True,
                                                                   drop_remainder=True)
        raw_val_dataset = tfrecord_datasets.get_tfrecord_dataset(val_records,
                                                                  self.schema.label_cols, batch_size,
                                                                  shuffle=False,
                                                                  drop_remainder=True)


        preprocess_config = preprocess.PreprocessingConfig(
            label_cols=self.schema.label_cols,
            input_size=initial_size,
            make_greyscale=self.greyscale,
            normalise_from_uint8=False  # False for tfrecords with 0-1 floats, True for png/jpg with 0-255 uints
        )
        train_dataset = preprocess.preprocess_dataset(raw_train_dataset, preprocess_config)
        val_dataset = preprocess.preprocess_dataset(raw_val_dataset, preprocess_config)

        with self.context_manager:
            model = define_model.get_model(
                output_dim=len(self.schema.label_cols),
                input_size=initial_size,
                crop_size=int(initial_size * 0.75),
                resize_size=resize_size,
                channels=self.channels,
                always_augment=always_augment,
                dropout_rate=dropout_rate
            )

            multiquestion_loss = losses.get_multiquestion_loss(self.schema.question_index_groups)
            # SUM reduction over loss, cannot divide by batch size on replicas when distributed training
            # so do it here instead
            loss = lambda x, y: multiquestion_loss(x, y) / batch_size
            # loss = multiquestion_loss
            
        model.compile(
            loss=loss,
            optimizer=tf.keras.optimizers.Adam(),
            # not happy about the next line
            # metrics=[tf.keras.metrics.CategoricalAccuracy()]
        )
        model.summary()

        train_config = training_config.TrainConfig(
            log_dir=self.save_dir,
            epochs=epochs,
            patience=10
        )

        # inplace on model
        training_config.train_estimator(
            model,
            train_config,  # parameters for how to train e.g. epochs, patience
            train_dataset,
            val_dataset,
            eager=eager  # set this True (or use --eager) for easier debugging, but slower training
        )
        elapsed = timedelta(seconds=(time() - start))
        logging.info(f"Finished training in {elapsed}")
        
def train_gz2_partial(params: dict):
    dataroot = Path(params['dataroot'])
    tm = TrainModel()
    tm.set_paths(shards_dir=str(dataroot / 'shards/gz2_partial'),
                 save_dir='results/gz2_partial')
    tm.set_schema('gz2_partial')
    tm.set_channels()
    tm.set_context_manager()
    tm.train(initial_size=32,
             resize_size=128,
             batch_size=8,
             epochs=2)

def train_gz2(params: dict, batch_size: int = 128, epochs: int = 100):
    dataroot = Path(params['dataroot'])
    tm = TrainModel()
    tm.set_paths(shards_dir=dataroot / 'shards/gz2',
                 save_dir=f'results/gz2/{datetime.now().strftime("%Y%m%d-%H%M%S")}')
    tm.set_schema('gz2')
    tm.set_channels()
    tm.set_context_manager()
    tm.train(initial_size=300,
             resize_size=128,
             batch_size=batch_size, # tried 128, ran out of memory locally
             epochs=epochs)

def train_decals(params: dict, batch_size: int = 128, epochs: int = 100):
    dataroot = Path(params['dataroot'])
    tm = TrainModel()
    tm.set_paths(shards_dir=dataroot / 'shards/decals',
                 save_dir=f'results/decals/{datetime.now().strftime("%Y%m%d-%H%M%S")}')
    tm.set_schema('decals')
    tm.set_channels()
    tm.set_context_manager()
    tm.train(initial_size=300,
             resize_size=224,
             batch_size=batch_size,
             epochs=epochs)


if __name__ == '__main__':
    # print(tf.__version__)
    params = read_params()
    train_gz2(params, batch_size=64, epochs=2)