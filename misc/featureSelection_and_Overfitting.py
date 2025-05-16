# %% [markdown]
# ### IMPORT

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import statsmodels.api as sm

f_sigmoid = lambda x: 1 / (1 + np.exp(-x))
n_samples = 300



for n_variables in [100, 500, 1000, 10000]:
    # SIMULATE
    n_variables_after_feature_selection = 10
    n_splits = 5
    x_data = np.random.randn(n_samples, n_variables)
    y_data = np.random.randn(n_samples, 1).flatten()
    y_data = f_sigmoid(y_data) >= 0.5
    
    p_values = []
    for col in range(n_variables):
        x = x_data[:, col].reshape(n_samples, 1)
        x = sm.add_constant(x)
        model = sm.Logit(y_data, x)
        results = model.fit(disp=False)
        p_values.append(results.pvalues[-1])

    p_values = np.array(p_values)
    p_values[np.argsort(p_values)]
    # Extract top 10 most significant features
    x_top10_features = x_data[:, np.argsort(p_values)][:, :n_variables_after_feature_selection]

    fig, ax = plt.subplots(1, n_splits, figsize=(10, 4))
    cv_splitter = KFold(n_splits=n_splits)
    probas_list = []
    true_y_list = []
    aucs = []
    cv_idx = 0
    for train_idx, test_idx in cv_splitter.split(x_top10_features, y_data):
        x_train, y_train = x_top10_features[train_idx], y_data[train_idx]
        x_test, y_test = x_top10_features[test_idx], y_data[test_idx]

        x_train = sm.add_constant(x_train)
        x_test = sm.add_constant(x_test)

        model = sm.Logit(y_train, x_train)
        results = model.fit(disp=False)

        test_proba = results.predict(x_test)
        test_auc = roc_auc_score(y_test, test_proba)
        x, y, _ = roc_curve(y_test, test_proba)
        ax[cv_idx].plot(x, y)
        ax[cv_idx].plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), linestyle="--", alpha=0.5)
        ax[cv_idx].set_xticks([])
        ax[cv_idx].set_yticks([])

        aucs.append(test_auc)
        probas_list += list(test_proba)
        true_y_list += list(y_test)

        cv_idx += 1

    plt.show()
    print(f"AUC test: {test_auc.mean()}")


    x, y, _ = roc_curve(true_y_list, probas_list)
    total_auc = roc_auc_score(true_y_list, probas_list)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(x, y)
    ax.set_title("ROC curve for CV-predictions after feature selection\non noise variables only")
    ax.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), linestyle="--", alpha=0.5)
    ax.text(0.3, 0.05, f"AUC for concattenated test predictions: {total_auc:.3}\nEXPERIMENT INFO\n\n{n_samples=}\n{n_variables=}\n{n_variables_after_feature_selection=}")
    # ax.text(0.4, 0.1, f"\nEXPERIMENT INFO\n{n_samples=}\n{n_varibles=}\n{n_variables_after_feature_selection=}")
    plt.show()

    print(f"Mean AUC across folds: {np.mean(aucs):.3}")
    print(f"\nEXPERIMENT INFO\n{n_samples=}, {n_variables=}, {n_variables_after_feature_selection=}")
