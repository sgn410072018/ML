from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from validation.classifiers import classifiers
from pathlib import Path
import numpy as np
import pandas as pd
import csv


class KFoldCrossValidator:

    def __init__(self, test_file_path):
        self._groups_file = "../train_data/groups.csv"
        self._X_train_file = test_file_path
        self._label_encoder = LabelEncoder()

    def get_accuracy_scores(self):
        X = self._parse_X()
        X = self._label_encode(X)

        groupsdata = self._parse_groups_data()
        groupsdata = self._label_encode(groupsdata)

        results = self.run_cross_validation(X, groupsdata, 10)
        sorted_results = sorted(results, key=lambda tup: tup[0])
        return sorted_results

    def run_cross_validation(self, X, groupsdata, folds):
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
        model = classifier
        model.fit(np.take(X, train, axis=0), np.take(y, train))
        pred = model.predict(np.take(X, test, axis=0))
        score = accuracy_score(np.take(y, test), pred, classifier)
        return score, classifier

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
        return data.astype(int)


if __name__ == "__main__":
    p = Path("../preprocessed-csv")
    csv_filenames = list(p.glob('*.csv'))
    with open('validation_results.csv', 'w') as file:
        total_scores = []
        for csv_filename in csv_filenames:
            filename_str = str(csv_filename)
            kFoldValidator = KFoldCrossValidator(filename_str)
            scores = kFoldValidator.get_accuracy_scores()
            scores = [(tup) + (filename_str,)for tup in scores]
            total_scores += scores

        writer = csv.writer(file, delimiter=';')
        writer.writerows(total_scores)





'''

X_test = np.load(X_test_file)
model = SVC(gamma="auto")
model.fit(X, y)
X_test_means = X_test.mean(2)
X_test_vars = X_test.var(2)
X_test = np.concatenate((X_test_means, X_test_vars), axis=1)
X_test = X_test[:, [4, 5, 6, 7, 8, 9, 14, 15, 16, 17]]
X_test = scaler.fit_transform(X_test)

y_pred = model.predict(X_test)
print(y_pred.astype(str))
labels = list(labelencoder.inverse_transform(y_pred))
print(labels)
with open("submission.csv", "w") as fp:
    fp.write("# Id,Surface\n")
    for i, label in enumerate(labels):
        fp.write("%d,%s\n" % (i, label))
'''