from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from validation.classifiers import classifiers
from pathlib import Path
import numpy as np
import pandas as pd
import csv
from itertools import groupby

# how many times the cross validation shuffles and folds the data
N_SHUFFLES_AND_FOLDS = 25

# file paths
GROUPS_FILE_PATH = "../train_data/groups.csv"
VALIDATION_RESULTS_FILE_PATH = "validation_results.csv"
PREPROCESSED_CSV_FILE_PATH = "../preprocessed-csv"


class KFoldCrossValidator:

    def __init__(self, test_file_path):
        self._groups_file = GROUPS_FILE_PATH
        self._X_train_file = test_file_path
        self._label_encoder = LabelEncoder()

    def get_accuracy_scores(self):
        X = self._parse_X()
        X = self._label_encode(X)
        X = X.astype(float)
        X = X[:, 2:]

        groupsdata = self._parse_groups_data()
        groupsdata = self._label_encode(groupsdata)
        groupsdata = groupsdata.astype(int)

        results = self.run_cross_validation(X, groupsdata)
        sorted_results = sorted(results, key=lambda tup: tup[0])
        return sorted_results

    def run_cross_validation(self, X, groupsdata, folds=N_SHUFFLES_AND_FOLDS):
        groups = groupsdata[:, 1]
        y = groupsdata[:, 2]
        gss = GroupShuffleSplit(random_state=1, n_splits=folds, test_size=0.2, train_size=0.8)
        results = []

        for train, test in gss.split(X, y, groups=groups):
            for classifier in classifiers:
                results.append(self._calc_model_accuracy(X, classifier, test, train, y))
        return results

    @staticmethod
    def _calc_model_accuracy(X, classifier, test, train, y):
        model = classifier[1]
        x_train = np.take(X, train, axis=0)
        y_train = np.take(y, train)
        model.fit(x_train, y_train)
        pred = model.predict(np.take(X, test, axis=0))
        score = accuracy_score(np.take(y, test), pred, classifier)
        return score, classifier[0]

    def _parse_X(self):
        X = pd.read_csv(self._X_train_file)
        X = X.values
        return X

    def _parse_groups_data(self):

        groupsdata = []
        with open(self._groups_file) as f:
            for line in f:
                values = line.split(",")
                values = [v.rstrip() for v in values]
                groupsdata.append(values)

        groupsdata = np.array(groupsdata[1:])
        return groupsdata

    def print_accuracy_scores(self):
        results = self.get_accuracy_scores()
        for result in results:
            print("{}\n{}\n\n".format(result[0], result[1]))

    def _label_encode(self, data):
        data[:, 2] = self._label_encoder.fit_transform(data[:, 2])
        return data


def main():
    p = Path(PREPROCESSED_CSV_FILE_PATH)
    csv_filenames = list(p.glob('*.csv'))
    with open(VALIDATION_RESULTS_FILE_PATH, 'w') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(['Accuracy', 'Classifier Definition', 'Source File'])
        total_scores = []
        for csv_filename in csv_filenames:
            total_scores += calc_classifier_scores_for_file(csv_filename)

        grouped_by_mean_and_file = group_by_mean_and_file(total_scores)

        writer.writerows(grouped_by_mean_and_file)


def calc_classifier_scores_for_file(csv_filename):
    filename_str = str(csv_filename)
    k_fold_validator = KFoldCrossValidator(filename_str)
    scores = k_fold_validator.get_accuracy_scores()
    scores = [tup + (filename_str,) for tup in scores]
    return scores


def group_by_mean_and_file(total_scores):
    grouped_mean_by_file = []
    total_scores = sorted(total_scores, key=lambda tup: tup[2])

    for k, g in groupby(total_scores, lambda tup: tup[2]):
        file_scores = sorted(list(g), key=lambda tup: tup[1])

        for k2, g2 in groupby(file_scores, lambda tup: tup[1]):
            grouped_mean_by_file.append((np.mean([cell[0] for cell in list(g2)]), k2, k))

    return grouped_mean_by_file


if __name__ == "__main__":
    main()
