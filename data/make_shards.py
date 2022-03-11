import logging
import pandas as pd
from pathlib import Path

from zoobot.data_utils import create_shards
from zoobot import label_metadata

from utils import read_params

class MakeShards:

    def __init__(self):

        logging.basicConfig(
            filename='make_shards.log',
            # filemode='w',
            format='%(asctime)s %(levelname)s:%(message)s',
            level=logging.INFO
        )

    def set_shard_type(self, shard_type):
        shard_types = ['gz2', 'gz2_partial', 'decals', 'decals_partial']
        if shard_type == 'gz2':
            self.label_cols = label_metadata.gz2_label_cols
        elif shard_type == 'gz2_partial':
            self.label_cols = label_metadata.gz2_partial_label_cols
        elif shard_type == 'decals':
            self.label_cols = label_metadata.decals_label_cols
        elif shard_type == 'decals_partial':
            self.label_cols = label_metadata.decals_partial_label_cols
        else:
            raise ValueError(f"shard_type {shard_type}, invalid, must be in {shard_types}")
        logging.info(f'Using {shard_type} label schema')

    def set_catalogs(self,
                     labelled_catalog_loc: str,
                     unlabelled_catalog_loc: str = '',
                     max_labelled: int = None,
                     max_unlabelled: int = None):

        # labels will always be floats, int conversion confuses tf.data
        dtypes = dict(zip(self.label_cols, [float for _ in self.label_cols]))
        dtypes['id_str'] = str
        self.labelled_catalog = pd.read_csv(labelled_catalog_loc, dtype=dtypes)
        if unlabelled_catalog_loc != '':
            self.unlabelled_catalog = pd.read_csv(unlabelled_catalog_loc, dtype=dtypes)
        else:
            self.unlabelled_catalog = None

        # limit catalogs to random subsets
        if max_labelled:
            self.labelled_catalog = self.labelled_catalog.sample(len(self.labelled_catalog))[:max_labelled]
        if max_unlabelled and (self.unlabelled_catalog is not None):
            self.unlabelled_catalog = self.unlabelled_catalog.sample(len(self.unlabelled_catalog))[:max_unlabelled]

        logging.info('Labelled catalog: {}'.format(len(self.labelled_catalog)))
        if self.unlabelled_catalog is not None:
            logging.info('Unlabelled catalog: {}'.format(len(self.unlabelled_catalog)))

    def make_shards(self,
                    shard_dir: str,
                    img_size: int,
                    eval_size: int = None):
        # in memory for now, but will be serialized for later/logs
        # train_test_fraction = create_shards.get_train_test_fraction(len(self.labelled_catalog), eval_size)

        labelled_columns_to_save = ['id_str'] + self.label_cols
        logging.info('Saving columns for labelled galaxies: \n{}'.format(labelled_columns_to_save))

        shard_config = create_shards.ShardConfig(shard_dir=shard_dir, size=img_size)

        # prepare_shards() defaults to train:test:eval of 0.7 : 0.2 : 0.1 - good enough for now?
        shard_config.prepare_shards(
            self.labelled_catalog,
            self.unlabelled_catalog,
            # test_fraction=train_test_fraction,
            labelled_columns_to_save=labelled_columns_to_save
        )


def gz2_partial_shards(labelled_catalog_loc: str,
                       shard_dir: str):
    ms = MakeShards()
    ms.set_shard_type('gz2_partial')
    ms.set_catalogs(labelled_catalog_loc=labelled_catalog_loc,
                    max_labelled=500,
                    max_unlabelled=300)
    ms.make_shards(shard_dir=shard_dir,
                   img_size=32,
                   eval_size=100)

def gz2_shards(labelled_catalog_loc: str,
                shard_dir: str):
    ms = MakeShards()
    ms.set_shard_type('gz2')
    ms.set_catalogs(labelled_catalog_loc=labelled_catalog_loc)
    ms.make_shards(shard_dir=shard_dir,
                   img_size=256,
                   eval_size=10000)


if __name__ == "__main__":
    params = read_params()
    dataroot = Path(params['dataroot'])
    gz2_partial_shards(str(dataroot / params['catalogs'] / 'gz2_partial_pairs.csv'),
                       str(dataroot / 'shards/gz2_partial'))
    gz2_shards(str(dataroot / params['catalogs'] / 'gz2_pairs.csv'),
               str(dataroot / 'shards/gz2'))