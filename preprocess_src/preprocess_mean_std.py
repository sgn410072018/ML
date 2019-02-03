import numpy as np
import pandas as pd


# df_train_y = pd.read_csv("../train_data/y_train_final_kaggle.csv")

#
TRAIN_X = "../train_data/X_train_kaggle.npy"
TEST_X = "../test_data/X_test_kaggle.npy"
TRAIN_Y = "../train_data/y_train_final_kaggle.csv"
GROUPS_ = "../train_data/groups.csv"
TRAIN_BEGIN = "../preprocessed-csv/groups_Xtrain_mean_std"
TRAIN_BEGIN_NORM = "../preprocessed-csv/groups_Xtrain_mean_std_normalized_0-1"
TEST_BEGIN = "../preprocessed-csv/test/groups_Xtest_mean_std"
TEST_BEGIN_NORM = "../preprocessed-csv/test/groups_Xtest_mean_std_normalized_0-1"

# measure_num = test_X.shape[2]


# TODO combinations
def get_mean_std(file_to_read):
    X = np.load(file_to_read)
    x_mean_std = []
    att_num = X.shape[1]
    case_num = X.shape[0]

    for a in range(att_num):
        this_attribute = []
        cur = X[:, a, :]

        for c in range(case_num):
            this_mean = np.mean(cur[c], axis=0)
            this_std = np.std(cur[c], axis=0)
            this_case = [this_mean, this_std]
            this_attribute.append(this_case)

        x_mean_std.append(this_attribute)
    return x_mean_std

# TODO the above with normalization

# todo alter to combinations
def combine_mean_std(x_mean_std, begin, cols = []):
    # create a pandas df for all the attributes separately. One df from one attribute and then later combine them as desired
    # for testing different attribute combinations
    dfs_mean_std = []

    colname = ""

    if cols == []:
        colname = ""

    df_groups = pd.read_csv(GROUPS_)


    # "Orientation_X"
    df_mean_std_orientationX = pd.DataFrame(x_mean_std[0],
                                                   columns=['OrientationX_mean', 'OrientationX_std', ])
    dfs_mean_std.append(df_mean_std_orientationX)

    # "Orientation_Y"
    df_mean_std_orientationY = pd.DataFrame(x_mean_std[1],
                                                   columns=['OrientationY_mean', 'OrientationY_std', ])
    dfs_mean_std.append(df_mean_std_orientationY)

    # "Orientation_Z"
    df_mean_std_orientationZ = pd.DataFrame(x_mean_std[2],
                                                   columns=['OrientationZ_mean', 'OrientationZ_std', ])
    dfs_mean_std.append(df_mean_std_orientationZ)

    # "Orientation_W"
    df_mean_std_orientationW = pd.DataFrame(x_mean_std[3],
                                                   columns=['OrientationW_mean', 'OrientationW_std', ])
    dfs_mean_std.append(df_mean_std_orientationW)

    # "AngularVelocity_X"
    df_mean_std_angularVelocityX = pd.DataFrame(x_mean_std[4],
                                                       columns=['AngularVelocityX_mean', 'AngularVelocityX_std', ])
    dfs_mean_std.append(df_mean_std_angularVelocityX)

    # "AngularVelocity_Y"
    df_mean_std_angularVelocityY = pd.DataFrame(x_mean_std[5],
                                                       columns=['AngularVelocityY_mean', 'AngularVelocityY_std', ])
    dfs_mean_std.append(df_mean_std_angularVelocityY)

    # "AngularVelocity_Z"
    df_mean_std_angularVelocityZ = pd.DataFrame(x_mean_std[6],
                                                       columns=['AngularVelocityZ_mean', 'AngularVelocityZ_std', ])
    dfs_mean_std.append(df_mean_std_angularVelocityZ)

    # "LinearAcceleration_X"
    df_mean_std_linearAccelerationX = pd.DataFrame(x_mean_std[7], columns=['LinearAccelerationX_mean',
                                                                                        'LinearAccelerationX_std', ])
    dfs_mean_std.append(df_mean_std_linearAccelerationX)

    # "LinearAcceleration_Y"
    df_mean_std_linearAccelerationY = pd.DataFrame(x_mean_std[8], columns=['LinearAccelerationY_mean',
                                                                                        'LinearAccelerationY_std', ])
    dfs_mean_std.append(df_mean_std_linearAccelerationY)

    # "LinearAcceleration_Z"
    df_mean_std_linearAccelerationZ = pd.DataFrame(x_mean_std[9], columns=['LinearAccelerationZ_mean',
                                                                                        'LinearAccelerationZ_std', ])
    dfs_mean_std.append(df_mean_std_linearAccelerationZ)

    df_mean_std = pd.concat(dfs_mean_std, axis=1, sort=False)
    df_groups_mean_std = pd.concat([df_groups, df_mean_std], axis=1, sort=False)

    # return df and the filename
    return df_groups_mean_std, begin + colname + ".csv"


def write_combined_mean_std(df, filename):
    df.to_csv(filename, index=False)

def main():
    x_mean_std = get_mean_std(TEST_X)
    df, filename = combine_mean_std(x_mean_std, TEST_BEGIN, [])
    write_combined_mean_std(df, filename)

if __name__ == "__main__":
    main()