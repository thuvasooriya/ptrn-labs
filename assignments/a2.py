"""
en3150 assignment 02: learning from data and related challenges and classification
instructor: sampath k. perera
author: thuvaragan s.
index no.: 210657G
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def listing1n2():
    df = sns.load_dataset("penguins")
    df.dropna(inplace=True)

    selected_classes = ["Adelie", "Chinstrap"]
    df_filtered = df[df["species"].isin(selected_classes)].copy()

    le = LabelEncoder()

    y_encoded = le.fit_transform(df_filtered["species"])

    df_filtered["class_encoded"] = y_encoded

    print(df_filtered[["species", "class_encoded"]])

    y = df_filtered["class_encoded"]
    X = df_filtered.drop(["species", "island", "sex", "class_encoded"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logreg = LogisticRegression(solver="saga")
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("saga accuracy:", accuracy)
    print(logreg.coef_, logreg.intercept_)

    logreg2 = LogisticRegression(solver="liblinear")
    logreg2.fit(X_train, y_train)
    y_pred2 = logreg2.predict(X_test)

    accuracy2 = accuracy_score(y_test, y_pred2)
    print("liblinear accuracy:", accuracy2)
    print(logreg2.coef_, logreg2.intercept_)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    logreg_liblinear = LogisticRegression(solver="liblinear")
    logreg_liblinear.fit(X_train_scaled, y_train)

    accuracy_liblinear = accuracy_score(y_test, logreg_liblinear.predict(X_test_scaled))
    print("liblinear accuracy with scaling:", accuracy_liblinear)
    print(logreg_liblinear.coef_, logreg2.intercept_)

    logreg_saga = LogisticRegression(solver="saga")
    logreg_saga.fit(X_train_scaled, y_train)

    accuracy_saga = accuracy_score(y_test, logreg_saga.predict(X_test_scaled))
    print("saga accuracy with scaling:", accuracy_saga)
    print(logreg_saga.coef_, logreg2.intercept_)


def listing3fixed():
    df = sns.load_dataset("penguins")
    df.dropna(inplace=True)

    selected_classes = ["Adelie", "Chinstrap"]
    df_filtered = df[
        df["species"].isin(selected_classes)
    ].copy()  # make a copy to avoid the warning

    le = LabelEncoder()
    y_encoded = le.fit_transform(df_filtered["species"])
    df_filtered["class_encoded"] = y_encoded

    print(df_filtered.head())

    # BUG:
    # X = df_filtered.drop(["species", "class_encoded"], axis=1)
    # fixed by dropping excess categorical data as in listing1
    X = df_filtered.drop(["species", "island", "sex", "class_encoded"], axis=1)
    y = df_filtered["class_encoded"]  # target variable
    # alternate fix - using one-hot encoding to encode other categorical features

    X.head()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logreg = LogisticRegression(solver="saga")
    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("accuracy:", accuracy)

    print(logreg.coef_, logreg.intercept_)


def lr_on_real_world_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    columns = ["ID", "Diagnosis"] + [f"feature_{i}" for i in range(1, 31)]
    data = pd.read_csv(url, header=None, names=columns)

    # map diagnosis to binary
    data["Diagnosis"] = data["Diagnosis"].map({"M": 1, "B": 0})

    unionset = data[["Diagnosis"] + [f"feature_{i}" for i in range(1, 31)]]

    # correlation matrix
    correlation_matrix = unionset.corr()
    fig1 = plt.figure(figsize=(24, 18))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", linewidths=0.5)
    plt.title("correlation matrix heatmap")
    fig1.savefig("./assignments/assets/a2q2fig1.png", bbox_inches="tight")
    # plt.show()

    # select up to 5 features for analysis
    features = ["feature_1", "feature_4", "feature_8", "feature_10", "feature_15"]
    subset = data[["Diagnosis"] + features]

    # pair plot using seaborn
    sns.pairplot(subset, hue="Diagnosis").savefig("./assignments/assets/a2q2fig2.png")
    # sns.pairplot(subset, hue="Diagnosis")
    # plt.show()

    X = subset[features]
    y = subset["Diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # fit logistic regression on scaled data
    model = LogisticRegression(solver="saga")
    # model = LogisticRegression(solver="liblinear")
    model.fit(X_train_scaled, y_train)
    # model.fit(X_train, y_train)

    # evaluate the model
    y_pred = model.predict(X_test_scaled)
    # y_pred = model.predict(X_test)

    # evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    cm_percentage = (
        conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
    )

    # Plot heatmap
    fig3 = plt.figure(figsize=(6, 5))
    sns.heatmap(cm_percentage, annot=True, fmt=".2f", cmap="Blues", cbar=False)
    # sns.heatmap(
    #     conf_matrix_percent,
    #     annot=True,
    #     fmt="d",
    #     # cmap="Blues",
    #     xticklabels=[1, 0],
    #     yticklabels=[1, 0],
    # )
    plt.xlabel("predicted")
    plt.ylabel("actual")
    plt.title("confusion matrix")
    fig3.savefig("./assignments/assets/a2q2fig3.png", bbox_inches="tight")
    plt.show()

    print(f"accuracy: {accuracy}")
    # print("confusion matrix:")
    # print(conf_matrix)
    print("classification report:")
    print(classification_rep)

    # fit logistic regression using statsmodels
    X_train_const = sm.add_constant(X_train)
    logit_model = sm.Logit(y_train, X_train_const)
    result = logit_model.fit()

    print(result.summary())


def listing4():
    # generate synthetic data
    np.random.seed(0)
    centers = [[-5, 0], [5, 1.5]]
    X, y = make_blobs(n_samples=2000, centers=centers, random_state=5)
    transformation = [[0.5, 0.5], [-0.5, 1.5]]
    X = np.dot(X, transformation)

    weights = np.random.randn(X.shape[1])
    learning_rate = 0.01
    iterations = 20

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def compute_loss(X, y, weights):
        z = np.dot(X, weights)
        predictions = sigmoid(z)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        # compute loss using clipped predictions
        loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        return loss

    def batch_gradient_descent(X, y, weights, learning_rate, iterations):
        m = len(y)
        losses = []

        for i in range(iterations):
            z = np.dot(X, weights)
            predictions = sigmoid(z)

            # gradient calculation
            gradient = np.dot(X.T, (predictions - y)) / m
            weights -= learning_rate * gradient

            # compute and store the loss
            loss = compute_loss(X, y, weights)
            losses.append(loss)

        return weights, losses

    # run batch gradient descent
    weights_bgd, losses_bgd = batch_gradient_descent(
        X, y, weights, learning_rate, iterations
    )

    def stochastic_gradient_descent(X, y, weights, learning_rate, iterations):
        m = len(y)
        losses = []

        for i in range(iterations):
            for j in range(m):
                rand_index = np.random.randint(0, m)
                x_i = X[rand_index : rand_index + 1]
                y_i = y[rand_index : rand_index + 1]
                z = np.dot(x_i, weights)
                predictions = sigmoid(z)

                gradient = np.dot(x_i.T, (predictions - y_i))
                weights -= learning_rate * gradient

            # Compute and store the loss
            loss = compute_loss(X, y, weights)
            losses.append(loss)

        return weights, losses

    # run sgd
    weights_sgd, losses_sgd = stochastic_gradient_descent(
        X, y, weights, learning_rate, iterations
    )

    def newton_method(X, y, weights, iterations):
        m = len(y)
        losses = []

        for i in range(iterations):
            z = np.dot(X, weights)
            predictions = sigmoid(z)

            gradient = np.dot(X.T, (predictions - y)) / m

            # hessian matrix
            diag = predictions * (1 - predictions)
            H = np.dot(X.T, diag[:, None] * X) / m

            # newton's update
            weights -= np.linalg.inv(H).dot(gradient)

            # compute and store the loss
            loss = compute_loss(X, y, weights)
            losses.append(loss)

        return weights, losses

    # run newton's method
    weights_newton, losses_newton = newton_method(X, y, weights, iterations)

    f1 = plt.figure(1)
    plt.plot(range(iterations), losses_bgd)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("loss vs epochs (batch gradient descent)")
    f1.savefig("./assignments/assets/a2q3fig1.png")
    # f1.show()

    f2 = plt.figure(2)
    plt.plot(range(iterations), losses_sgd)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("loss vs epochs (stochastic gradient descent)")
    f2.savefig("./assignments/assets/a2q3fig2.png")
    # f2.show()

    f3 = plt.figure(3)
    plt.plot(range(iterations), losses_newton)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("loss vs epochs (newton method)")
    f3.savefig("./assignments/assets/a2q3fig3.png")
    # f3.show()

    f4 = plt.figure(4)
    plt.plot(range(iterations), losses_bgd, label="batch gradient descent")
    plt.plot(range(iterations), losses_sgd, label="stochastic gradient descent")
    plt.plot(range(iterations), losses_newton, label="newton's method")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.title("loss vs epochs for different methods")
    f4.savefig("./assignments/assets/a2q3fig4.png")
    # f4.show()

    # change the centers
    print("changing centers")
    centers_new = [[3, 0], [5, 1.5]]
    X_new, y_new = make_blobs(n_samples=2000, centers=centers_new, random_state=5)
    X_new = np.dot(X_new, transformation)

    # apply batch gradient descent again
    weights_new, losses_new = batch_gradient_descent(
        X_new, y_new, weights, learning_rate, iterations
    )

    f5, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.set_title("loss vs epochs (batch gradient decent - new centers)")
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("loss")
    ax1.legend()
    ax1.plot(range(iterations), losses_new, label="bgd (new center)")
    ax2.set_title("loss vs epochs (batch gradient decent - old & new)")
    ax2.set_xlabel("epochs")
    ax2.set_ylabel("loss")
    ax2.legend()
    ax2.plot(range(iterations), losses_new, label="bgd (new center)")
    ax2.plot(range(iterations), losses_bgd, label="bgd (initial)")
    plt.tight_layout()
    f5.savefig("./assignments/assets/a2q3fig5.png")
    plt.show()

    f6, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=10)
    ax1.set_title("Old Data (centers = [[-5, 0], [5, 1.5]])")
    ax1.set_xlabel("Feature 1")
    ax1.set_ylabel("Feature 2")
    ax2.scatter(X_new[:, 0], X_new[:, 1], c=y_new, cmap="coolwarm", s=10)
    ax2.set_title("New Data (centers = [[3, 0], [5, 1.5]])")
    ax2.set_xlabel("Feature 1")
    ax2.set_ylabel("Feature 2")
    plt.tight_layout()
    f6.savefig("./assignments/assets/a2q3fig6.png")
    plt.show()


if __name__ == "__main__":
    # uncomment the section of the assignment you want to test and run the file
    # of course all can be tested at the same time
    # listing1n2()
    # listing3()
    lr_on_real_world_data()
    # listing4()
