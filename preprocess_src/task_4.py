from preprocess_src import preprocess as p

# path for the preprocessed csvfiles
PATH_TRAIN = "../preprocessed-csv/"
PATH_TEST = "../preprocessed-csv/test/"

X_train = p.ravelled(p.get_train_X())
filename = "task_4_ravelled_.csv"
p.write_train_csv(X_train, PATH_TRAIN + filename)

X_test = p.ravelled(p.get_test_X())
filename = "task_4_ravelled_test.csv"
p.write_test_csv(X_test, PATH_TEST + filename)


X_train = p.meaned(p.get_train_X())
filename = "task_4_meaned_.csv"
p.write_train_csv(X_train, PATH_TRAIN + filename)

X_test = p.meaned(p.get_test_X())
filename = "task_4_meaned_test.csv"
p.write_test_csv(X_test, PATH_TEST + filename)


X_train = p.stdied(p.get_train_X())
filename = "task_4_stdied_.csv"
p.write_train_csv(X_train, PATH_TRAIN + filename)

X_test = p.stdied(p.get_test_X())
filename = "task_4_stdied_test.csv"
p.write_test_csv(X_test, PATH_TEST + filename)


X_train = p.mean_std(p.get_train_X())
filename = "task_4_mean_std_.csv"
p.write_train_csv(X_train, PATH_TRAIN + filename)

X_test = p.mean_std(p.get_test_X())
filename = "task_4_mean_std_test.csv"
p.write_test_csv(X_test, PATH_TEST + filename)


X_train = p.accelerations_velocities_abs(p.get_train_X())
filename = "task_4_accelerations_velocities_abs_.csv"
p.write_train_csv(X_train, PATH_TRAIN + filename)

X_test = p.accelerations_velocities_abs(p.get_test_X())
filename = "task_4_accelerations_velocities_abs_test.csv"
p.write_test_csv(X_test, PATH_TEST + filename)


X_train = p.mean_std(X_train)
filename = "task_4_accelerations_velocities_abs_mean_std_.csv"
p.write_train_csv(X_train, PATH_TRAIN + filename)

X_test = p.mean_std(X_test)
filename = "task_4_accelerations_velocities_abs_mean_std_test.csv"
p.write_test_csv(X_test, PATH_TEST + filename)

