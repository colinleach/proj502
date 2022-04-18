from pathlib import Path
import numpy as np
import pandas as pd
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

    @staticmethod
    def summarize(df: pd.DataFrame, output: pd.DataFrame):
        """

        :param df:
        :param output:
        :return:
        """
        for question in decals_pairs:
            options = decals_pairs[question]
            pairs = [question + opt for opt in options]
            df_subset = df.loc[:, pairs]
            summary = pd.DataFrame()
            summary['idxmax'] = ((df_subset.idxmax(axis=1)).str.split('_'))
            summary['maxprob'] = df_subset.max(axis=1, numeric_only=True) \
                                 / df_subset.sum(axis=1, numeric_only=True)
            summary['choice'] = summary.values.tolist()
            output[question] = summary['choice']
        return output

    def make_prediction_summaries(self):
        """

        :return:
        """
        summaries = self.ids.copy()
        self.summaries = self.summarize(self.results, summaries)

    def make_observed_summaries(self, catalog_file: Path) -> None:
        """

        :param catalog_file:
        :return:
        """
        self.obs = pd.read_csv(catalog_file)
        summaries = pd.DataFrame(self.obs['file_loc'], columns=['file_loc', ])
        self.obs_summaries = self.summarize(self.obs, summaries)

    def one_comparison(self, inx: int) -> pd.DataFrame:
        """

        :param inx:
        :return:
        """

        example = self.summaries.iloc[inx]
        questions = self.summaries.columns.values[1:]
        preds = [f"{q}: {example[q][0][1]}, ({example[q][1]:.2f})" for q in questions]
        obs = self.obs_summaries.loc[self.obs['file_loc'] == example['file_loc']].iloc[0]

        comp = pd.DataFrame(columns=['question', 'pred', 'pred_confidence', 'obs', 'obs_confidence'])
        comp['question'] = questions
        comp['pred'] = [example[q][0][1] for q in questions]
        comp['pred_confidence'] = [f"{example[q][1]:.2f}" for q in questions]
        comp['obs'] = [obs[q][0][1] for q in questions]
        comp['obs_confidence'] = [obs[q][1] for q in questions]
        return example['file_loc'], comp


if __name__ == "__main__":
    params = read_params()
    dataroot = Path(params['dataroot'])
    save_loc = dataroot / 'results/predictions/decals.hdf5'
    catalog_file = dataroot / 'shards/decals/test_shards/test_df.csv'
    vp = ViewPredictions()
    vp.read_predictions(save_loc)
    vp.make_prediction_summaries()
    vp.make_observed_summaries(catalog_file)
    comp = vp.one_comparison(20)
    print(comp)