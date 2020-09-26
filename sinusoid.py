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
parser.add_argument('--time', type=int, help="number of timesteps")
parser.add_argument('--samples', type=int, help="number of samples for initialization")
parser.add_argument('--kernel', type=int, help="kernel size")
parser.add_argument('--channels', type=int, help="number of channels")
parser.add_argument('--layers', type=int, help="number of layers")
parser.add_argument('--eta', type=float, help='learning rate')
parser.add_argument('--eps', type=float, help='eps squared')
parser.add_argument('--relu', type=bool, default=False, help="use relu in lifting")
parser.add_argument('--data', type=int, help="choice of dataset")
parser.add_argument('--lmbda', type=float, default=5000, help='lambda')
parser.add_argument('--tail', type=str, default='',
                    help='extra information to add to folder name')
parser.add_argument('--save_dir', type=str, default='./saved_models/',
                    help='base directory for saving PyTorch model. (default: ./saved_models/)')
parser.add_argument('--data_dir', type=str, default='./data/mnist',
                    help='base directory for saving PyTorch model. (default: ./data/mnist/)')
args = parser.parse_args()

model_dir = os.path.join("./saved_models", "wave_fourier", f"data{args.data}",
                         "time{}_samples{}_kernel{}_channels{}_layers{}_eps{}_eta{}_relu{}"
                         "".format(args.time, args.samples, args.kernel, args.channels,
                                   args.layers, args.eps, args.eta, args.relu))
os.makedirs(model_dir, exist_ok=True)

# data loading
X_train, y_train, num_classes = dataset.generate_wave(args.time, args.samples, args.data, shuffle=True)
X_test, y_test, _ = dataset.generate_wave(args.time, args.samples, args.data, shuffle=False)
X_translate, y_translate, _ = dataset.generate_wave(args.time, 1, args.data, shuffle=False, augment=True)

# setup architecture
kernels = dataset.generate_kernel('random', args.channels, 1, args.kernel)
layers = [Lift1D(kernels, relu=args.relu)] + [Fourier1D(args.layers, eta=args.eta, eps=args.eps)]
model = Architecture(layers, model_dir, num_classes)

# train/test pass
print("Forward pass - train features")
Z_train = model(X_train, y_train).real
utils.save_loss(model.loss_dict, model_dir, "init")
utils.save_loss(model.loss_dict, model_dir, "train")
print("Forward pass - test features")
Z_test = model(X_test).real
utils.save_loss(model.loss_dict, model_dir, "test")
print("Forward pass - translated features")
Z_translate = model(X_translate).real
utils.save_loss(model.loss_dict, model_dir, "translate")

# save features
utils.save_features(model_dir, "Z_train", Z_train, y_train)
utils.save_features(model_dir, "Z_test", Z_test, y_test)
utils.save_features(model_dir, "Z_translate", Z_translate, y_test)
np.save(os.path.join(model_dir, "features", "kernel.npy"), kernels)

X_train = tf.normalize(X_train.reshape(X_train.shape[0], -1))
X_test = tf.normalize(X_test.reshape(X_test.shape[0], -1))
X_translate = tf.normalize(X_translate.reshape(X_translate.shape[0], -1))
Z_train = tf.normalize(Z_train.reshape(Z_train.shape[0], -1))
Z_test = tf.normalize(Z_test.reshape(Z_test.shape[0], -1))
Z_translate = tf.normalize(Z_translate.reshape(Z_translate.shape[0], -1))

# evaluation test
_, acc_svm = evaluate.svm(Z_train, y_train, Z_test, y_test)
acc_knn = evaluate.knn(Z_train, y_train, Z_test, y_test, k=5)
acc_svd = evaluate.nearsub(Z_train, y_train, Z_test, y_test, n_comp=5)
acc = {"svm": acc_svm, "knn": acc_knn, "nearsub-svd": acc_svd} 
utils.save_params(model_dir, acc, name="acc_test.json")

# evaluation translate
_, acc_svm = evaluate.svm(Z_train, y_train, Z_translate, y_translate)
acc_knn = evaluate.knn(Z_train, y_train, Z_translate, y_translate, k=5)
acc_svd = evaluate.nearsub(Z_train, y_train, Z_translate, y_translate, n_comp=5)
acc = {"svm": acc_svm, "knn": acc_knn, "nearsub-svd": acc_svd} 
utils.save_params(model_dir, acc, name="acc_translate.json")

# plot
plot.plot_combined_loss(model_dir)
plot.plot_combined_loss_translation(model_dir)
plot.plot_heatmap(X_train, y_train, "X_train", num_classes, model_dir)
plot.plot_heatmap(X_test, y_test, "X_test", num_classes, model_dir)
plot.plot_heatmap(X_translate, y_translate, "X_translate", num_classes, model_dir)
plot.plot_heatmap(Z_train, y_train, "Z_train", num_classes, model_dir)
plot.plot_heatmap(Z_test, y_test, "Z_test", num_classes, model_dir)
plot.plot_heatmap(Z_translate, y_translate, "Z_translate", num_classes, model_dir)
