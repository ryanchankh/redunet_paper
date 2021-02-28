import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from redunet import Architecture, Vector
import dataset
import evaluate
import plot
import utils

# hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--layers', type=int, default=20, help="number of layers")
parser.add_argument('--eta', type=float, default=0.5, help='learning rate')
parser.add_argument('--eps', type=float, default=0.1, help='eps squared')
parser.add_argument('--tail', type=str, default='',
                    help='extra information to add to folder name')
parser.add_argument('--save_dir', type=str, default='./saved_models/',
                    help='base directory for saving PyTorch model. (default: ./saved_models/)')
parser.add_argument('--data_dir', type=str, default='./data/',
                    help='base directory for saving PyTorch model. (default: ./saved_models/)')
args = parser.parse_args()

# pipeline setup
model_dir = os.path.join("./saved_models", "mice",
                         "layers{}_eps{}_eta{}"
                         "".format(args.layers, args.eps, args.eta)
                         )
os.makedirs(model_dir, exist_ok=True)
utils.save_params(model_dir, vars(args))
print(model_dir)

# data setup
X_train, y_train, X_test, y_test, num_classes = dataset.load_Mice(args.data_dir)

# model setup
layers = [Vector(args.layers, eta=args.eta, eps=args.eps)]
model = Architecture(layers, model_dir, num_classes)

# train/test pass
print("Forward pass - train features")
Z_train = model(X_train, y_train)
utils.save_loss(model.loss_dict, model_dir, "train")
print("Forward pass - test features")
Z_test = model(X_test)
utils.save_loss(model.loss_dict, model_dir, "test")

# save features
utils.save_features(model_dir, "X_train", X_train, y_train)
utils.save_features(model_dir, "X_test", X_test, y_test)
utils.save_features(model_dir, "Z_train", Z_train, y_train)
utils.save_features(model_dir, "Z_test", Z_test, y_test)

# evaluation
_, acc_svm = evaluate.svm(Z_train, y_train, Z_test, y_test)
acc_knn = evaluate.knn(Z_train, y_train, Z_test, y_test, k=5)
acc_svd = evaluate.nearsub(Z_train, y_train, Z_test, y_test, n_comp=5)
acc = {"svm": acc_svm, "knn": acc_knn, "nearsub-svd": acc_svd} 
utils.save_params(model_dir, acc, name="acc_test.json")

# plot
plot.plot_combined_loss(model_dir)
plot.plot_heatmap(X_train, y_train, "X_train", model_dir)
plot.plot_heatmap(X_test, y_test, "X_test", model_dir)
plot.plot_heatmap(Z_train, y_train, "Z_train", model_dir)
plot.plot_heatmap(Z_test, y_test, "Z_test", model_dir)

