from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.naive_bayes import GaussianNB

threads = 4

classifiers= [
    KNeighborsClassifier(n_jobs=threads, n_neighbors=20),
    KNeighborsClassifier(n_jobs=threads, n_neighbors=50),
    KNeighborsClassifier(n_jobs=threads, n_neighbors=100),
    KNeighborsClassifier(n_jobs=threads, n_neighbors=100, weights="distance"),
    LinearDiscriminantAnalysis(),
    LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto"),
    LinearDiscriminantAnalysis(solver="svd"),
    LinearDiscriminantAnalysis(solver="lsqr"),
    SVC(gamma="auto"),
    SVC(gamma=0.1),
    SVC(gamma=1),
    SVC(gamma=10),
    SVC(gamma=30),
    SVC(kernel="sigmoid", gamma="auto"),
    SVC(degree=1, gamma="auto"),
    SVC(degree=2, gamma="auto"),
    LogisticRegression(multi_class='auto', solver='liblinear', intercept_scaling=5),
    LogisticRegression(n_jobs=threads, multi_class='auto', solver='newton-cg', max_iter=1000),
    LogisticRegression(n_jobs=threads, class_weight="balanced", penalty="l2", multi_class="auto", solver="lbfgs"),
    LogisticRegression(n_jobs=threads, penalty="l2", multi_class="auto", solver="lbfgs"),
    RandomForestClassifier(n_jobs=threads, n_estimators=20),
    RandomForestClassifier(n_jobs=threads, n_estimators=50),
    RandomForestClassifier(n_jobs=threads, n_estimators=100),
    RandomForestClassifier(n_jobs=threads, n_estimators=100, max_depth=5),
    AdaBoostClassifier(n_estimators=20),
    AdaBoostClassifier(n_estimators=50),
    ExtraTreesClassifier(n_jobs=threads, n_estimators=20),
    ExtraTreesClassifier(n_jobs=threads, n_estimators=50),
    ExtraTreesClassifier(n_jobs=threads, n_estimators=50, criterion="entropy"),
    GaussianNB()
]
