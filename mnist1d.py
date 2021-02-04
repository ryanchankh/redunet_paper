import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from redunet import (
    Architecture, 
    Fourier1D, 
    Lift1D,
)
import dataset
import evaluate
import plot
import functionals as F
import utils

# hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--samples', type=int, help="number of samples for initialization")
parser.add_argument('--channels', type=int, help="number of channels")
parser.add_argument('--outchannels', type=int, help="number of outchannels")
parser.add_argument('--ksize', type=int, help="kernel size")
parser.add_argument('--time', type=int, help="number of timesteps")
parser.add_argument('--classes', nargs="+", type=int, help="Classes to Keep (Example: 0 1)")
parser.add_argument('--layers', type=int, help="number of layers")
parser.add_argument('--eta', type=float, help='learning rate')
parser.add_argument('--eps', type=float, help='eps squared')
parser.add_argument('--lmbda', type=float, default=500, help='lambda')
parser.add_argument('--tail', type=str, default='',
                    help='extra information to add to folder name')
parser.add_argument('--save_dir', type=str, default='./saved_models/',
                    help='base directory for saving PyTorch model. (default: ./saved_models/)')
args = parser.parse_args()

model_dir = os.path.join(args.save_dir, f"mnist1d-classes{''.join(map(str, args.classes))}",
                         "time{}_channels{}_samples{}_ksize{}_outchannels{}_layers{}_eps{}_eta{}{}"
                         "".format(args.time, args.channels, args.samples, args.ksize, args.outchannels,
                                   args.layers, args.eps, args.eta, args.tail))
os.makedirs(model_dir, exist_ok=True)
utils.save_params(model_dir, vars(args))

# data loading
X_train, y_train = dataset.load_MNIST("./data/mnist/", train=True)
X_test, y_test = dataset.load_MNIST("./data/mnist", train=False)
X_train, y_train = F.filter_class(X_train, y_train, args.classes, args.samples)
X_train = F.convert2polar(X_train, args.channels, args.time)
X_train, y_train = F.shuffle(X_train, y_train)
X_test, y_test = F.filter_class(X_test, y_test, args.classes, args.samples)
X_test = F.convert2polar(X_test, args.channels, args.time)
X_translate_train, y_translate_train = F.translate1d(*F.get_n_each(X_train, y_train, 10), n=2, stride=4)
X_translate_test, y_translate_test = F.translate1d(*F.get_n_each(X_test, y_test, 10), n=2, stride=4)

# save features
utils.save_features(model_dir, "X_train", X_train, y_train)
utils.save_features(model_dir, "X_test", X_test, y_test)
utils.save_features(model_dir, "X_translate_train", X_translate_train, y_translate_train)
utils.save_features(model_dir, "X_translate_test", X_translate_test, y_translate_test)

# setup architecture
kernels = np.random.normal(0, 1, size=(args.outchannels, args.channels, args.ksize))
np.save(os.path.join(model_dir, 'features', 'kernel.npy'), kernels)
layers = [Lift1D(kernels)] + [Fourier1D(args.layers, eta=args.eta, eps=args.eps)]
model = Architecture(layers, model_dir, len(args.classes))

# train/test pass
print("Forward pass - train features")
Z_train = model(X_train, y_train).real
utils.save_loss(model.loss_dict, model_dir, "train")
print("Forward pass - test features")
Z_test = model(X_test).real
utils.save_loss(model.loss_dict, model_dir, "test")
print("Forward pass - translate train features")
Z_translate_train = model(X_translate_train).real
utils.save_loss(model.loss_dict, model_dir, "train")
print("Forward pass - translate test features")
Z_translate_test = model(X_translate_test).real
utils.save_loss(model.loss_dict, model_dir, "test")

# save features
utils.save_features(model_dir, "Z_train", Z_train, y_train)
utils.save_features(model_dir, "Z_test", Z_test, y_test)
utils.save_features(model_dir, "Z_translate_train", Z_translate_train, y_translate_train)
utils.save_features(model_dir, "Z_translate_test", Z_translate_test, y_translate_test)

# normalize
X_train = F.normalize(X_train.reshape(X_train.shape[0], -1))
X_test = F.normalize(X_test.reshape(X_test.shape[0], -1))
X_translate_train = F.normalize(X_translate_train.reshape(X_translate_train.shape[0], -1))
X_translate_test = F.normalize(X_translate_test.reshape(X_translate_test.shape[0], -1))
Z_train = F.normalize(Z_train.reshape(Z_train.shape[0], -1))
Z_test = F.normalize(Z_test.reshape(Z_test.shape[0], -1))
Z_translate_train = F.normalize(Z_translate_train.reshape(Z_translate_train.shape[0], -1))
Z_translate_test = F.normalize(Z_translate_test.reshape(Z_translate_test.shape[0], -1))

# plot
plot.plot_combined_loss(model_dir)
plot.plot_heatmap(X_train, y_train, "X_train", model_dir)
plot.plot_heatmap(X_test, y_test, "X_test", model_dir)
plot.plot_heatmap(X_translate_train, y_translate_train, "X_translate_train", model_dir)
plot.plot_heatmap(X_translate_test, y_translate_test, "X_translate_test", model_dir)
plot.plot_heatmap(Z_train, y_train, "Z_train", model_dir)
plot.plot_heatmap(Z_test, y_test, "Z_test", model_dir)
plot.plot_heatmap(Z_translate_train, y_translate_train, "Z_translate_train", model_dir)
plot.plot_heatmap(Z_translate_test, y_translate_test, "Z_translate_test", model_dir)

# evaluation test
_, acc_svm = evaluate.svm(Z_train, y_train, Z_test, y_test)
acc_knn = evaluate.knn(Z_train, y_train, Z_test, y_test, k=5)
acc_svd = evaluate.nearsub(Z_train, y_train, Z_test, y_test, n_comp=20)
acc = {"svm": acc_svm, "knn": acc_knn, "nearsub-svd": acc_svd} 
utils.save_params(model_dir, acc, name="acc_test.json")

# evaluation translate train
_, acc_svm = evaluate.svm(Z_train, y_train, Z_translate_train, y_translate_train)
acc_knn = evaluate.knn(Z_train, y_train, Z_translate_train, y_translate_train, k=5)
acc_svd = evaluate.nearsub(Z_train, y_train, Z_translate_train, y_translate_train, n_comp=20)
acc = {"svm": acc_svm, "knn": acc_knn, "nearsub-svd": acc_svd} 
utils.save_params(model_dir, acc, name="acc_translate_train.json")

# evaluation translate test
_, acc_svm = evaluate.svm(Z_train, y_train, Z_translate_test, y_translate_test)
acc_knn = evaluate.knn(Z_train, y_train, Z_translate_test, y_translate_test, k=5)
acc_svd = evaluate.nearsub(Z_train, y_train, Z_translate_test, y_translate_test, n_comp=20)
acc = {"svm": acc_svm, "knn": acc_knn, "nearsub-svd": acc_svd} 
utils.save_params(model_dir, acc, name="acc_translate_test.json")

