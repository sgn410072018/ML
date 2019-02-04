import numpy as np
import pandas as pd

# files
TRAIN_X = "../train_data/X_train_kaggle.npy"
TEST_X = "../test_data/X_test_kaggle.npy"
TRAIN_Y = "../train_data/y_train_final_kaggle.csv"
GROUPS_ = "../train_data/groups.csv"


# some functions for getting data
def get_train_X():
    return np.load(TRAIN_X)


def get_test_X():
    return np.load(TEST_X)


def get_train_Y_dataframe():
    return pd.read_csv(TRAIN_Y, delimiter=",")


def get_train_Y_numpy():
    return pd.read_csv(TRAIN_Y, delimiter=",").values


def get_train_Y_list():
    return pd.read_csv(TRAIN_Y, delimiter=",").values.tolist()


def get_groups_dataframe():
    return pd.read_csv(GROUPS_, delimiter=",")


def get_groups_numpy():
    return pd.read_csv(GROUPS_, delimiter=",").values


def get_groups_list():
    return pd.read_csv(GROUPS_, delimiter=",").values.tolist()


# functions for preprocessing
# 4.a) Straightforward reshape usingnumpy.ravel()ornumpy.resize()tomake 1280-dimensional vector from each sample.
def ravelled(numpy_data):
    rav = []
    samples = numpy_data.shape[0]
    for sample in range(samples):
        rav.append(np.ravel(numpy_data[sample,:,:]))
    return np.array(rav) #, "ravelled"


# b) Compute the average over the time axis usingnumpy.mean()with ap-propriateaxisargument. This should make each sample 10-dimensional
def meaned(numpy_data):
    return np.mean(numpy_data, axis=2) #, "means taken"


def stdied(numpy_data):
    return np.std(numpy_data, axis=2) #, "std taken"


# c) Compute the average and standard deviation over the time axis and con-catenate these to 20-dimensional feature vectors.
def mean_std(numpy_data):
    return np.concatenate((meaned(numpy_data), stdied(numpy_data)), axis=1) #, "mean and std"


# d)  Choose which sensors you wish to use (remove those not wanted)
# sensors_to_remove of unique numbers 0..9 eg [1, 2, 3, 5, 8]
def remove_sensors(numpy_data, sensors_to_remove):
    return np.delete(numpy_data, sensors_to_remove, axis=1) #, "removed sensors " + str(sensors_to_remove)


# accelerations and velocities to absolute values
# deletes orientation values and returns only the rest
def accelerations_velocities_abs(numpy_data):
    numpy_data = remove_sensors(numpy_data, [0,1,2,3])
    return np.array(list(map(abs, numpy_data))) #, "acc and vel absed"


# write to file for validation part
def write_train_csv(numpy_data, filename):
    if len(numpy_data.shape) == 3:
        df_data = pd.DataFrame(ravelled(numpy_data))
    else:
        df_data = pd.DataFrame(numpy_data)
    df_groups = get_groups_dataframe()

    df = pd.concat([df_groups, df_data], axis=1, sort=False)
    df.to_csv(filename, index=False)


# write to file for guessing part
def write_test_csv(numpy_data, filename):
    if len(numpy_data.shape) == 3:
        numpy_data = ravelled(numpy_data)

    pd.DataFrame(numpy_data).to_csv(filename, index=False)
