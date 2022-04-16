from pathlib import Path
import numpy as np
import pandas as pd
import os
import h5py

from projcode.data.utils import read_params
from zoobot.shared.label_metadata import decals_pairs

class ViewPredictions:

    # def __init__(self, save_loc: Path, feather_file: Path = None):
    #     if feather_file is None:
    #         self.read_predictions(save_loc)
    #     else:
    #         self.read_feather(feather_file)
    #     self.make_summaries()

    def read_predictions(self, save_loc):
        with h5py.File(save_loc, "r") as f:
            id_str = [id.decode("utf-8") for id in list(f['id_str'])]
            label_cols = [l.decode("utf-8") for l in list(f['label_cols'])]
            predictions = list(f['predictions'])

        means = [np.mean(p, axis=1) for p in predictions]
        npmeans = np.stack(means)
        preds = pd.DataFrame(npmeans, columns=label_cols)
        self.ids = pd.DataFrame(id_str, columns=['file_loc', ])

        self.results = pd.concat([self.ids, preds], axis=1)

    def read_feather(self, feather_file: Path):
        self.results = pd.read_feather(feather_file)
        self.ids = self.results.loc('file_loc')

    def write_feather(self, feather_file: Path):
        self.results.to_feather(feather_file)

    def make_summaries(self):
        summaries = self.ids.copy()
        for question in decals_pairs:
            options = decals_pairs[question]
            pairs = [question + opt for opt in options]
            result_subset = self.results.loc[:, pairs]
            summary = pd.DataFrame()
            summary['idxmax'] = ((result_subset.idxmax(axis=1)).str.split('_'))
            summary['maxprob'] = result_subset.max(axis=1, numeric_only=True) \
                                 / result_subset.sum(axis=1, numeric_only=True)
            summary['choice'] = summary.values.tolist()
            summaries[question] = summary['choice']
        self.summaries = summaries


if __name__ == "__main__":
    params = read_params()
    dataroot = Path(params['dataroot'])
    save_loc = dataroot / 'results/predictions/decals.hdf5'
    vp = ViewPredictions()
    vp.read_predictions(save_loc)
    vp.make_summaries()