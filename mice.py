import argparse
import os

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from arch import (
    Architecture, 
    Vector
)
import dataset
import evaluate
import plot
import train_func as tf
import utils

# hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--layers', type=int, help="number of layers")
parser.add_argument('--eta', type=float, help='learning rate')
parser.add_argument('--eps', type=float, help='eps squared')
parser.add_argument('--lmbda', type=float, default=5000, help='lambda')
parser.add_argument('--tail', type=str, default='',
                    help='extra information to add to folder name')
parser.add_argument('--save_dir', type=str, default='./saved_models/',
                    help='base directory for saving PyTorch model. (default: ./saved_models/)')
args = parser.parse_args()

# pipeline setup
model_dir = os.path.join("./saved_models", "mice_approxinit",
                         "layers{}_eps{}_eta{}"
                         "".format(args.layers, args.eps, args.eta)
                         )
os.makedirs(model_dir, exist_ok=True)
utils.save_params(model_dir, vars(args))
print(model_dir)

# data setup
df_data = pd.read_csv('./data/mice/data.csv')
df_data = df_data.fillna(df_data.mean())
df_data['class'] = df_data['class'].astype('category').cat.codes
X = df_data.to_numpy()[:, 1:78].astype(np.float32)
y = df_data['class'].to_numpy().astype(np.int32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_init, y_init = X_train, y_train
num_classes = 8

# model setup
layers = [Vector(args.layers, eta=args.eta, eps=args.eps, lmbda=args.lmbda)]
model = Architecture(layers, model_dir, num_classes)

Z_init = model(X_init, y_init).real
utils.save_loss(model.loss_dict, model_dir, "init")

# train/test pass
print("Forward pass - train features")
Z_train = model(X_train).real
utils.save_loss(model.loss_dict, model_dir, "train")
print("Forward pass - test features")
Z_test = model(X_test).real
utils.save_loss(model.loss_dict, model_dir, "test")

# save features
utils.save_features(model_dir, "X_train", X_train, y_train)
utils.save_features(model_dir, "X_test", X_test, y_test)
utils.save_features(model_dir, "Z_train", Z_train, y_train)
utils.save_features(model_dir, "Z_test", Z_test, y_test)

X_train = tf.normalize(X_train)
X_test = tf.normalize(X_test)
Z_train = tf.normalize(Z_train)
Z_test = tf.normalize(Z_test)

# evaluation
_, acc_svm = evaluate.svm(Z_train, y_train, Z_test, y_test)
acc_knn = evaluate.knn(Z_train, y_train, Z_test, y_test, k=5)
acc_svd = evaluate.nearsub(Z_train, y_train, Z_test, y_test, n_comp=1)
acc = {"svm": acc_svm, "knn": acc_knn, "nearsub-svd": acc_svd} 
utils.save_params(model_dir, acc, name="acc_test.json")

# plot
plot.plot_combined_loss(model_dir)
plot.plot_heatmap(X_train, y_train, "X_train", model_dir)
plot.plot_heatmap(X_test, y_test, "X_test", model_dir)
plot.plot_heatmap(Z_train, y_train, "Z_train", model_dir)
plot.plot_heatmap(Z_test, y_test, "Z_test", model_dir)

# comparison
# from sklearn.linear_model import SGDClassifier
# from sklearn.svm import LinearSVC, SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier

# test_models = {'log_l2': SGDClassifier(loss='log', max_iter=10000, random_state=42),
#                'SVM_linear': LinearSVC(max_iter=10000, random_state=42),
#                'SVM_RBF': SVC(kernel='rbf', random_state=42),
#                'DecisionTree': DecisionTreeClassifier(),
#                'RandomForrest': RandomForestClassifier()}

# for model_name in test_models:
#     test_model = test_models[model_name]
#     test_model.fit(X_train, y_train)
#     score = test_model.score(X_test, y_test)
#     print(f"{model_name}: {score}")

