import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from arch import (
    Architecture, 
    Fourier1D, 
    Lift1D,
)
import dataset
import evaluate
import plot
import train_func as tf
import utils

# hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--samples', type=int, help="number of samples for initialization")
parser.add_argument('--classes', nargs="+", type=int, help="Classes to Keep (Example: 0 1)")
parser.add_argument('--layers', type=int, help="number of layers")
parser.add_argument('--eta', type=float, help='learning rate')
parser.add_argument('--eps', type=float, help='eps squared')
parser.add_argument('--lmbda', type=float, default=5000, help='lambda')
parser.add_argument('--tail', type=str, default='',
                    help='extra information to add to folder name')
parser.add_argument('--save_dir', type=str, default='./saved_models/',
                    help='base directory for saving PyTorch model. (default: ./saved_models/)')
args = parser.parse_args()

model_dir = os.path.join(args.save_dir, f"mnist_rotation-classes{''.join(map(str, args.classes))}",
                         "samples{}_layers{}_eps{}_eta{}{}"
                         "".format(args.samples, args.layers, args.eps, args.eta, args.tail))
os.makedirs(model_dir, exist_ok=True)
utils.save_params(model_dir, vars(args))

# data loading
# X_train, y_train = dataset.load_MNIST_polar("./data/mnist/", args.samples, args.channels, args.time, train=True)
# X_test, y_test = dataset.load_MNIST_polar("./data/mnist/", args.samples, args.channels, args.time, train=False)
X_train = np.load('./data/mnist_rotation/mnist_X_train_C5_T200.npy')
y_train = np.load('./data/mnist_rotation/mnist_y_train_C5_T200.npy')
X_test = np.load('./data/mnist_rotation/mnist_X_test_C5_T200.npy')
y_test = np.load('./data/mnist_rotation/mnist_y_test_C5_T200.npy')
X_train, y_train = tf.filter_class(X_train, y_train, args.classes, args.samples)
X_train, y_train = tf.shuffle(X_train, y_train)
X_test, y_test = tf.filter_class(X_test, y_test, args.classes, args.samples)
X_each, y_each = tf.get_n_each(X_test, y_test, 20)
X_translate, y_translate = tf.translate_all(X_each, y_each, stride=5)

# setup architecture
layers = [Fourier1D(args.layers, eta=args.eta, eps=args.eps)]
model = Architecture(layers, model_dir, len(args.classes))

# train/test pass
print("Forward pass - train features")
Z_train = model(X_train, y_train).real
utils.save_loss(model.loss_dict, model_dir, "train")
print("Forward pass - test features")
Z_test = model(X_test).real
utils.save_loss(model.loss_dict, model_dir, "test")
print("Forward pass - translated features")
Z_translate = model(X_translate).real
utils.save_loss(model.loss_dict, model_dir, "translate")

# save features
utils.save_features(model_dir, "X_train", X_train, y_train)
utils.save_features(model_dir, "X_test", X_test, y_test)
utils.save_features(model_dir, "X_translate", X_translate, y_translate)
utils.save_features(model_dir, "Z_train", Z_train, y_train)
utils.save_features(model_dir, "Z_test", Z_test, y_test)
utils.save_features(model_dir, "Z_translate", Z_translate, y_translate)

X_train = tf.normalize(X_train.reshape(X_train.shape[0], -1))
X_test = tf.normalize(X_test.reshape(X_test.shape[0], -1))
X_translate = tf.normalize(X_translate.reshape(X_translate.shape[0], -1))
Z_train = tf.normalize(Z_train.reshape(Z_train.shape[0], -1))
Z_test = tf.normalize(Z_test.reshape(Z_test.shape[0], -1))
Z_translate = tf.normalize(Z_translate.reshape(Z_translate.shape[0], -1))

# evaluation test
_, acc_svm = evaluate.svm(Z_train, y_train, Z_test, y_test)
acc_knn = evaluate.knn(Z_train, y_train, Z_test, y_test, k=5)
acc_svd = evaluate.nearsub(Z_train, y_train, Z_test, y_test, n_comp=100)
acc = {"svm": acc_svm, "knn": acc_knn, "nearsub-svd": acc_svd} 
utils.save_params(model_dir, acc, name="acc_test.json")

# evaluation translate
_, acc_svm = evaluate.svm(Z_train, y_train, Z_translate, y_translate)
acc_knn = evaluate.knn(Z_train, y_train, Z_translate, y_translate, k=5)
acc_svd = evaluate.nearsub(Z_train, y_train, Z_translate, y_translate, n_comp=200)
acc = {"svm": acc_svm, "knn": acc_knn, "nearsub-svd": acc_svd} 
utils.save_params(model_dir, acc, name="acc_translate.json")

# plot
plot.plot_combined_loss(model_dir)
plot.plot_heatmap(X_train, y_train, "X_train", model_dir)
plot.plot_heatmap(X_test, y_test, "X_test", model_dir)
plot.plot_heatmap(X_translate, y_translate, "X_translate", model_dir)
plot.plot_heatmap(Z_train, y_train, "Z_train", model_dir)
plot.plot_heatmap(Z_test, y_test, "Z_test", model_dir)
plot.plot_heatmap(Z_translate, y_translate, "Z_translate", model_dir)
plot.plot_nearsub_angle(X_train, y_train, Z_train, X_translate, y_translate, Z_translate, 200, model_dir)
