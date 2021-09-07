import os

import numpy as np
import pandas as pd

from scorings import get_scoring


class Results():

    def __init__(self, data_type, data_name, mode, dirpath="results"):
        self.root_path = dirpath
        self.__ensure_root_path__()

        self.scorings_path = os.path.join(self.root_path, "scorings")
        self.__ensure_scorings_path__()
        self.scores_table_path = os.path.join(
            self.scorings_path,
            self.__correct_file_name__(f"{data_type}_{data_name}_{mode}.csv")
        )

        self.predictions_path = os.path.join(self.root_path, "predictions")
        self.__ensure_predictions_path__()

        self.scores_table_df = None
        self.data_type = data_type
        self.data_name = data_name
        assert mode in ["test", "val"]
        self.mode = mode

        self.models_preds = {}
        self.models_scores = {}
        self.__load_scores_table__()
        self.unset_model()

    def __ensure_root_path__(self):
        if not os.path.isdir(self.root_path):
            os.mkdir(self.root_path)

    def __ensure_scorings_path__(self):
        self.__ensure_root_path__()
        if not os.path.isdir(self.scorings_path):
            os.mkdir(self.scorings_path)

    def __ensure_predictions_path__(self):
        self.__ensure_root_path__()
        if not os.path.isdir(self.predictions_path):
            os.mkdir(self.predictions_path)

    def __correct_file_name__(self, filename):
        return filename.replace("/", " ")

    def __save_scores_table__(self):
        if self.scores_table_df.values.size != 0:
            self.__ensure_scorings_path__()
            self.scores_table_df.to_csv(self.scores_table_path)

    def __mode_name__(self):
        return self.mode[0].upper() + self.mode[1:]

    def __load_scores_table__(self):
        if os.path.exists(self.scores_table_path):
            self.scores_table_df = pd.read_csv(self.scores_table_path,
                                               header=[0], index_col=[0, 1])
            self.scores_table_df.columns = self.scores_table_df.columns.set_names(
                ["Scoring"])
        else:
            self.scores_table_df = pd.DataFrame(
                data=[],
                index=pd.MultiIndex.from_arrays([[]] * 2).set_names(["Model", "Version"]),
                columns=pd.Index([]).set_names(["Metric"]),
            )
            table_title = f"{self.data_type}: {self.data_name} – Mean {self.__mode_name__()} Scores of Models"
            self.scores_table_df.style.set_caption(table_title)

    def __add_model__(self, model_name, model_version):
        if (model_name, model_version) not in self.scores_table_df.index:
            self.scores_table_df.loc[(model_name, model_version), :] = np.nan
#             self.scores_table_df = self.scores_table_df.append(
#                 pd.Series([], name=(model_name, model_version)))

    def __add_scoring__(self, scoring_name):
        if scoring_name not in self.scores_table_df.columns:
            self.scores_table_df[scoring_name] = np.nan

    def set_model(self, model_name, model_version):
        self.model_name = model_name
        self.model_version = model_version

    def unset_model(self):
        self.model_name = None
        self.model_version = None

    def get_model(self):
        return dict(model_name=self.model_name,
                    model_version=self.model_version)

    def scores_table(self):
        self.__load_scores_table__()
        return self.scores_table_df

    def reorder_models(self, models_names):
        assert set(self.scores_table_df.index.values) == set(models_names)
        self.__load_scores_table__()
        self.scores_table_df = self.scores_table_df.reindex(models_names)
        self.__save_scores_table__()

    def reorder_scorings(self, scorings_names):
        assert set(self.scores_table_df.columns.values) == set(scorings_names)
        self.__load_scores_table__()
        self.scores_table_df = self.scores_table_df[scorings_names]
        self.__save_scores_table__()

    def __get_model_scores_path__(self, path):
        return os.path.join(
            path,
            self.__correct_file_name__(
                f"{self.model_name}_{self.model_version}.npy")
        )

    def __add_dir_to_path_and_ensure__(self, path, dirname):
        path = os.path.join(path, self.__correct_file_name__(dirname))
        if not os.path.isdir(path):
            os.mkdir(path)
        return path

    def add_model_scores(self, all_scores, mean_scores=None):
        assert self.model_name is not None
        assert self.model_version is not None

        if self.model_name not in self.models_scores:
            self.models_scores[self.model_name] = {}
        self.models_scores[self.model_name][self.model_version] = all_scores

        self.__ensure_scorings_path__()
        path = self.scorings_path
        path = self.__add_dir_to_path_and_ensure__(path, self.data_type)
        path = self.__add_dir_to_path_and_ensure__(path, self.data_name)
        path = self.__add_dir_to_path_and_ensure__(path, self.mode)
        np.save(self.__get_model_scores_path__(path), all_scores)

        if mean_scores is None:
            mean_scores = self.calc_mean_scores(all_scores)

        self.__load_scores_table__()
        self.__add_model__(self.model_name, self.model_version)
        scorings_names = mean_scores.keys()
        for scoring_name in scorings_names:
            self.__add_scoring__(scoring_name)
            self.scores_table_df.loc[(self.model_name, self.model_version), scoring_name] = \
                mean_scores[scoring_name]
        self.__save_scores_table__()

    def __add_dir_to_path_and_raise__(self, path, dirname):
        path = os.path.join(path, self.__correct_file_name__(dirname))
        if not os.path.isdir(path):
            raise Exception(f"Dictionary '{path}' does not exist")
        return path

    def load_model_scores(self):
        assert self.model_name is not None
        assert self.model_version is not None

        path = self.scorings_path
        path = self.__add_dir_to_path_and_raise__(path, self.data_type)
        path = self.__add_dir_to_path_and_raise__(path, self.data_name)
        path = self.__add_dir_to_path_and_ensure__(path, self.mode)
        all_scores = np.load(self.__get_model_scores_path__(path),
                             allow_pickle="TRUE").item()

        if self.model_name not in self.models_scores:
            self.models_scores[self.model_name] = {}
        self.models_scores[self.model_name][self.model_version] = all_scores

    def get_model_scores(self):
        assert self.model_name is not None
        assert self.model_version is not None
        return self.models_scores[self.model_name][self.model_version]

    def add_model_preds(self, preds):
        assert self.model_name is not None
        assert self.model_version is not None
        if self.model_name not in self.models_preds:
            self.models_preds[self.model_name] = {}
        self.models_preds[self.model_name][self.model_version] = preds

        self.__ensure_predictions_path__()
        path = self.predictions_path

        for dirname in [
            self.data_type,
            self.data_name,
            self.mode,
            self.model_name,
            self.model_version
        ]:
            path = self.__add_dir_to_path_and_ensure__(path, dirname)

        for i, pred in enumerate(preds):
            number = f"{i}"
            number = "0" * (4 - len(number)) + number
            begin_idx = f"{pred.index[0]}"
            end_idx = f"{pred.index[-1]}"
            pred.to_csv(
                os.path.join(
                    path,
                    f"{number} ({self.__correct_file_name__(begin_idx)} - {self.__correct_file_name__(end_idx)}).csv",
                )
            )

    def load_model_preds(self):
        assert self.model_name is not None
        assert self.model_version is not None
        self.__ensure_predictions_path__()
        path = self.predictions_path

        for dirname in [
            self.data_type,
            self.data_name,
            self.mode,
            self.model_name,
            self.model_version
        ]:
            path = self.__add_dir_to_path_and_raise__(path, dirname)

        preds = []
        parent_path = path
        for filename in sorted(os.listdir(parent_path)):
            path = os.path.join(parent_path, filename)
            s = pd.read_csv(path, index_col=0, parse_dates=True, squeeze=True)
            preds.append(s)

        if self.model_name not in self.models_preds:
            self.models_preds[self.model_name] = {}
        self.models_preds[self.model_name][self.model_version] = preds

    def get_model_preds(self):
        assert self.model_name is not None
        assert self.model_version is not None
        return self.models_preds[self.model_name][self.model_version]

    def calc_all_scores(self, true_ts, preds, scorings_names):
        all_scores = {}
        for scoring_name in scorings_names:
            scoring = get_scoring(scoring_name)
            all_scores[scoring_name] = []
            for pred in preds:
                score = scoring(true_ts.loc[pred.index], pred)
                all_scores[scoring_name].append(score)
        return all_scores

    def calc_mean_scores(self, all_scores):
        mean_scores = {}
        for scoring_name in all_scores.keys():
            mean_scores[scoring_name] = np.mean(all_scores[scoring_name])
        return mean_scores

# def get_data_index():
#     global ts_data
#     data = [
#         (data_type, data)
#         for data_type in ts_data.keys()
#         for data in ts_data[data_type].keys()
#     ]
#     if len(data) == 0:
#         index = pd.MultiIndex.from_arrays([[]] * 2)
#     else:
#         index = pd.Index(
#             data=[
#                 (data_type, data)
#                 for data_type in ts_data.keys()
#                 for data in ts_data[data_type].keys()
#             ]
#         )
#     return index.set_names(["Data Type", "Data"])
