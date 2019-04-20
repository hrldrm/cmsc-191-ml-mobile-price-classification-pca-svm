from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from pathlib import Path
from time import gmtime, strftime, time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import getopt
import sys

NUMERIC_VARS = ["battery_power", "clock_speed", "fc", "int_memory", "m_dep", "mobile_wt",
                "n_cores", "pc", "px_height", "px_width", "ram", "sc_h", "sc_w", "talk_time"]

dataset_path = Path("../mobile-price-classification/")


def classify_data(train_data, test_data, train_label, test_label):
    clf = svm.SVC(gamma=0.001, kernel="linear")
    clf.fit(train_data, train_label)
    pred = clf.predict(test_data)
    return accuracy_score(test_label, pred, normalize=True), \
        precision_score(test_label, pred, average="micro"), f1_score(
            test_label, pred, average="micro")


def decompose_data(train_data, test_data, components):
    pca_decomposer = PCA(n_components=components)
    pca_decomposer.fit(train_data)
    return pca_decomposer.transform(train_data), \
        pca_decomposer.transform(test_data), \
        pca_decomposer.transform(train_data).shape, \
        pca_decomposer.explained_variance_ratio_.cumsum()


def scale_data(train_data, test_data):
    scaler = StandardScaler()
    scaler.fit(train_data)
    return scaler.transform(train_data), scaler.transform(test_data)


def load_data():
    print("--- Loading Data ---")
    time_start = time()
    df = pd.read_csv(
        dataset_path / "train.csv"
    )
    X_train, X_test, y_train, y_test = train_test_split(
        df.iloc[:, :-1], df["price_range"], test_size=0.3, random_state=42)
    time_taken = strftime("%H:%M:%S", gmtime(time() - time_start))
    print("Loading data took {}".format(time_taken))
    print(len(df.columns))
    return X_train, X_test, y_train, y_test, np.arange(1, len(df.columns))


if __name__ == "__main__":
    train_data, test_data, train_target, test_target, components_range = load_data()
    train_data, test_data = scale_data(train_data, test_data)
    log_df = pd.DataFrame(
        columns=["pca_n_components", "shape", "accuracy", "precision", "f1_score", "cumsum"])

    for components in components_range:
        train_data_dec, test_data_dec, shape, var_cumsum = decompose_data(
            train_data, test_data, components)

        print("Cumulative variance with n_components = {0}: {1:.4%}".format(
            components, var_cumsum[-1]))

        acc_res, prec_res, f1_res = classify_data(train_data_dec, test_data_dec,
                                                  train_target, test_target)
        log_df = log_df.append(pd.Series({"pca_n_components": components, "shape": str(shape), "accuracy": acc_res,
                                          "precision": prec_res, "f1_score": f1_res, "cumsum": var_cumsum}), ignore_index=True)
        log_df.to_csv("log.csv")
    log_df.plot(x="pca_n_components", y="accuracy", kind="line")
    plt.xlabel("Number of Components")
    plt.show()
    plt.plot(log_df.loc[:, "pca_n_components"], log_df.ix[len(
        log_df.index) - 1, "cumsum"], 'g-')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.show()
