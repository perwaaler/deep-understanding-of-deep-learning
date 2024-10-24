"""Different ways of splitting the data into train and test sets using pandas,
"""

# %%
import torch
import seaborn as sns
import path_setup
from utilities import *
import numpy as np
from classes import IrisModel, ModelTrainer

# %%
df = sns.load_dataset("iris")

data = torch.tensor(
    df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values
).float()
labels = df["species"].replace({"setosa": 0, "versicolor": 1, "virginica": 2})


orange_print(f"Label Counts:\n{labels.value_counts()}")
labels = torch.tensor(labels)

propTraining = 0.7
nTraining = int(len(data) * propTraining)


index_training = np.random.choice(data.shape[0], nTraining, replace=False)
index_training_bool = df.index.isin(index_training)
index_test_bool = ~df.index.isin(index_training)


np.array(len(data), dtype=bool)

magenta_print(f"Training set: {index_training_bool}")

# %%
# torch.mean(labels[index_training])

model = IrisModel()
trainer = ModelTrainer(model)

trainer.train_model(data[index_training_bool], labels[index_training_bool])
trainer.plot_losses()

# %% Test

y_hat_test = trainer.model(data[index_test_bool]).argmax(axis=1)
test_accuracy = torch.mean((y_hat_test == labels[index_test_bool]).float()).item()

green_print(f"Test accuracy: {100*test_accuracy}")


# %% ALternative using Pandas sampling
orange_print("USING PANDAS FOR SAMPLING")
df = sns.load_dataset("iris")
import pandas as pd

pd.set_option("future.no_silent_downcasting", True)

df_train = df.sample(frac=0.8, replace=False)
df_test = df.loc[df.index.difference(df_train.index)]
green_print(
    f"Confirm Independence",
    f"Intersection between sets: {df_test.index.intersection(df_train.index)}",
)


x_vars = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def iris_df_to_tensor(df):
    data = torch.tensor(df[x_vars].values).float()
    labels = (
        df["species"]
        .replace({"setosa": 0, "versicolor": 1, "virginica": 2})
        .astype(int)
    )
    labels = torch.tensor(labels.values)
    return data, labels


data_train, labels_train = iris_df_to_tensor(df=df_train)
data_test, labels_test = iris_df_to_tensor(df_test)

model = IrisModel()
trainer = ModelTrainer(model)
trainer.train_model(data_train, labels_train)
trainer.plot_losses()

scores_0 = trainer.model(data_test)[0]

y_hat = trainer.model(data_test).argmax(axis=1)
torch.mean((y_hat == labels_test).float()).item()


# %% Using scikit learn
from sklearn.model_selection import train_test_split
import path_setup
from classes import IrisModel, ModelTrainer
import seaborn as sns
import torch

x_vars = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

df = sns.load_dataset("iris")

X_all = df[x_vars].values
y_all = (
    df["species"].replace({"setosa": 0, "versicolor": 1, "virginica": 2}).astype(int)
).values

x_vars = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.1)

X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)


model = IrisModel()
trainer = ModelTrainer(model, n_epochs=200)
model = trainer.train_model(
    x_train=X_train, labels_train=y_train, x_valid=X_test, labels_valid=y_test
)

# plt.plot(trainer.valid_accuracy)

# trainer.valid_accuracy
# trainer.accuracy_valid.item()

# %% Usin Data Loader
import torch

# DataLoader is an iterator
from torch.utils.data import DataLoader, TensorDataset
import path_setup
from classes import ModelTrainer, IrisModel
import seaborn as sns
from utilities import *
from sklearn.model_selection import train_test_split

n_samples = 100
n_nodes_input = 3
n_nodes_output = 4
sigma_e = 10

# Weights matrix that maps X to Y
W = torch.randn(n_nodes_input, n_nodes_output)
X = torch.randn(n_samples, n_nodes_input)
noise = torch.randn(n_samples, n_nodes_output)
Y = torch.matmul(X, W) + noise * sigma_e
labels = Y.argmax(axis=1)

x_train, x_test, y_train, y_test = train_test_split(X, labels)

model = IrisModel()
modelTrainer = ModelTrainer(model)
modelTrainer.test_training()
# y_data = torch.random.

fakeDatasetTraining = TensorDataset(x_train, y_train)
fakeDatasetTraining = DataLoader(fakeDatasetTraining, batch_size=10)

fakeDatasetValidation = TensorDataset(x_test, y_test)
fakeDatasetValidation = DataLoader(fakeDatasetValidation)

for i, (x, y) in enumerate(fakeDatasetTraining):
    red_print(f"batch: {i}")
    green_print(x.detach()[:10])
    orange_print(x.shape)

magenta_print(f"Batches: {i}")
teal_print(f"YOU ARE DONE!!!")
pink_print(f"YOU ARE DONE!!!")
