import argparse
import os

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from arch import (
    Architecture, 
    Lift1D,
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
parser.add_argument('--noise', type=float, help='noise')
parser.add_argument('--lmbda', type=float, default=5000, help='lambda')
parser.add_argument('--data', type=int, help='choice of distributions for data')
parser.add_argument('--tail', type=str, default='',
                    help='extra information to add to folder name')
parser.add_argument('--save_dir', type=str, default='./saved_models/',
                    help='base directory for saving PyTorch model. (default: ./saved_models/)')
args = parser.parse_args()

# pipeline setup
model_dir = os.path.join(args.save_dir, "iris", "layers{}_eps{}_eta{}"
                         "".format(args.layers, args.eps, args.eta))
os.makedirs(model_dir, exist_ok=True)
utils.save_params(model_dir, vars(args))

# data setup
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.3)
X_train = tf.normalize(X_train)
X_test = tf.normalize(X_test)
num_classes = 3
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# model setup
layers = [Vector(args.layers, eta=args.eta, eps=args.eps, lmbda=args.lmbda)]
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
acc_svd = evaluate.nearsub(Z_train, y_train, Z_test, y_test, n_comp=3)
acc = {"svm": acc_svm, "knn": acc_knn, "nearsub": acc_svd} 
utils.save_params(model_dir, acc, name="acc_test.json")

# plot
plot.plot_combined_loss(model_dir)
plot.plot_heatmap(X_train, y_train, "X_train", model_dir)
plot.plot_heatmap(X_test, y_test, "X_test", model_dir)
plot.plot_heatmap(Z_train, y_train, "Z_train", model_dir)
plot.plot_heatmap(Z_test, y_test, "Z_test", model_dir)
# save per layers
# layers_dir = os.path.join(model_dir, "features", "layers")
# for filename in os.listdir(layers_dir):
#     filepath = os.path.join(layers_dir, filename)
#     Z = np.load(filepath)
#     plot.plot_2d(Z, y_init, filename[:-4], model_dir)
