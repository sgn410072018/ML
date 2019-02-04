from validation.classifiers import classifiers
from os import path
from pathlib import Path
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

# file paths and names
VALIDATION_RESULTS_FILE_PATH = "../validation"   # path to validation results files
TRAIN_FILE_PATH = "../preprocessed-csv"  # path to training data files
TEST_FILE_PATH = "../preprocessed-csv/test/"  # path to test data files

SUBMISSION_FILE_BEGIN = "submission_files/submission_"


# find the best validation result and make a submission using the info it contains
def submissionFromBest(val_results_path, trainfile_path, testfile_path, sub_file_begin):

    best_val_res = bestValidationResult(val_results_path)

    # get the classifier, train data and test data corresponding to the best validation result
    # there exists a unique corresponding test data file for every train data file.
    clf = getClassifier(best_val_res[1], classifiers)
    X, y = getTrainData(trainfile_path, best_val_res[2])
    X_test = getTestX(testfile_path + testFromTrain(best_val_res[2]))

    # fit the classifier with all of the train data
    le = LabelEncoder()
    clf.fit(X, le.fit_transform(y))

    # predict with the test data
    y_pred = list(le.inverse_transform(clf.predict(X_test)))
    # print(len(y_pred))

    # create a submission filename from standard beginning and datetime
    submission_filename = submissionFileName(sub_file_begin, best_val_res)

    writeSubmissionFile(y_pred, submission_filename)



# find the best result from validation results
# if many with same accuracy in some file, this chooses the first encountered from that file.
def bestValidationResult(filepath):
    best_acc = 0
    p = Path(filepath)
    filenames = list(p.glob('*.csv'))
    for filename in filenames:
        df = pd.read_csv(filename, delimiter=";")
        i = df["Accuracy"].idxmax()
        this = df.iloc[i]
        this_acc = this["Accuracy"]
        if this_acc > best_acc:
            best = this
            best_acc = best["Accuracy"]
    return best.to_list()


# list of classifiers is from classifiers.py
def getClassifier(clf_definition, clfs):
    for clf in clfs:
        if clf[0] == clf_definition:
            return clf[1]


def getTrainData(filepath, filename):
    p = Path(filepath)
    filenames = list(p.glob('*.csv'))
    df = 0

    for fname in filenames:
        if str(fname) == filename:
            df = pd.read_csv(fname, delimiter=",")
            break

    y = df['Surface'].values #[1:]
    df = df.drop(['# Id', 'Group Id', 'Surface'], axis=1)
    X = df.values
    return X, y


def testFromTrain(trainfilepath):
    fname = path.basename(trainfilepath)
    train_split = fname.split("Xtrain")
    return train_split[0] + "Xtest" + train_split[1]


def getTestX(filename):
    df = pd.read_csv(filename, delimiter=",")
    return df.drop(['# Id', 'Group Id', 'Surface'], axis=1).values


def submissionFileName(begin, best):
    dt = datetime.now().strftime("_%Y-%m-%d-%H:%M")
    b1 = str(round(best[0], 3)).split(".")[1]
    b2 = "_" + str(best[1])
    return begin + b1 + b2 + dt + ".csv"


def writeSubmissionFile(y, filename):
    with open(filename, "w") as f:
        f.write("# Id,Surface\n")
        for i, label in enumerate(y):
            f.write("%d,%s\n" % (i, label))


def main():
    submissionFromBest(VALIDATION_RESULTS_FILE_PATH, TRAIN_FILE_PATH, TEST_FILE_PATH, SUBMISSION_FILE_BEGIN)


if __name__ == "__main__":
    main()
