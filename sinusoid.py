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
parser.add_argument('--data', type=int, required=True, help='choice of distributions for data')
parser.add_argument('--time', type=int, default=100, help="number of timesteps")
parser.add_argument('--outchannels', type=int, default=10, help="number of channels")
parser.add_argument('--ksize', type=int, default=9, help="kernel size")
parser.add_argument('--samples', type=int, default=20, help="number of samples")
parser.add_argument('--layers', type=int, default=10, help="number of layers")
parser.add_argument('--eta', type=float, default=0.5, help='learning rate')
parser.add_argument('--eps', type=float, default=0.1, help='eps squared')
parser.add_argument('--tail', type=str, default='',
                    help='extra information to add to folder name')
parser.add_argument('--save_dir', type=str, default='./saved_models/',
                    help='base directory for saving PyTorch model. (default: ./saved_models/)')
args = parser.parse_args()

model_dir = os.path.join("./saved_models", "sinusoid", f"data{args.data}",
                         f"time{args.time}"
                         f"_samples{args.samples}"
                         f"_outchannels{args.outchannels}"
                         f"_layers{args.layers}"
                         f"_eps{args.eps}"
                         f"_eta{args.eta}"
                         f"{args.tail}")
os.makedirs(model_dir, exist_ok=True)
utils.save_params(model_dir, vars(args))
print(model_dir)

# data loading
X_train, y_train, num_classes = dataset.generate_wave(args.time, args.samples, args.data, shuffle=True)
X_test, y_test, _ = dataset.generate_wave(args.time, args.samples, args.data, shuffle=False)
X_translate_train, y_translate_train = F.translate1d(*F.get_n_each(X_train, y_train, 10), n=2, stride=4)
X_translate_test, y_translate_test = F.translate1d(*F.get_n_each(X_test, y_test, 10), n=2, stride=4)

# setup architecture
kernels = F.generate_kernel('gaussian', (args.outchannels, 1, args.ksize))
layers = [Lift1D(kernels)] + [Fourier1D(args.layers, eta=args.eta, eps=args.eps)]
model = Architecture(layers, model_dir, num_classes)

# train/test pass
print("Forward pass - train features")
Z_train = model(X_train, y_train).real
utils.save_loss(model.loss_dict, model_dir, "train")
print("Forward pass - test features")
Z_test = model(X_test).real
utils.save_loss(model.loss_dict, model_dir, "test")
print("Forward pass - translated train features")
Z_translate_train = model(X_translate_train).real
utils.save_loss(model.loss_dict, model_dir, "translate_train")
print("Forward pass - translated test features")
Z_translate_test = model(X_translate_test).real
utils.save_loss(model.loss_dict, model_dir, "translate_test")

# save features
utils.save_features(model_dir, "X_train", X_train, y_train)
utils.save_features(model_dir, "X_test", X_test, y_test)
utils.save_features(model_dir, "X_translate_train", X_translate_train, y_translate_train)
utils.save_features(model_dir, "X_translate_test", X_translate_test, y_translate_test)
utils.save_features(model_dir, "Z_train", Z_train, y_train)
utils.save_features(model_dir, "Z_test", Z_test, y_test)
utils.save_features(model_dir, "Z_translate_train", Z_translate_train, y_translate_train)
utils.save_features(model_dir, "Z_translate_test", Z_translate_test, y_translate_test)
np.save(os.path.join(model_dir, 'features', 'kernel.npy'), kernels)

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

# evaluation train
_, acc_svm = evaluate.svm(Z_train, y_train, Z_train, y_train)
acc_knn = evaluate.knn(Z_train, y_train, Z_train, y_train, k=5)
acc_svd = evaluate.nearsub(Z_train, y_train, Z_train, y_train, n_comp=10)
acc = {"svm": acc_svm, "knn": acc_knn, "nearsub-svd": acc_svd} 
utils.save_params(model_dir, acc, name="acc_train.json")

# evaluation test
_, acc_svm = evaluate.svm(Z_train, y_train, Z_test, y_test)
acc_knn = evaluate.knn(Z_train, y_train, Z_test, y_test, k=5)
acc_svd = evaluate.nearsub(Z_train, y_train, Z_test, y_test, n_comp=10)
acc = {"svm": acc_svm, "knn": acc_knn, "nearsub-svd": acc_svd} 
utils.save_params(model_dir, acc, name="acc_test.json")

# evaluation translate train
_, acc_svm = evaluate.svm(Z_train, y_train, Z_translate_train, y_translate_train)
acc_knn = evaluate.knn(Z_train, y_train, Z_translate_train, y_translate_train, k=5)
acc_svd = evaluate.nearsub(Z_train, y_train, Z_translate_train, y_translate_train, n_comp=10)
acc = {"svm": acc_svm, "knn": acc_knn, "nearsub-svd": acc_svd} 
utils.save_params(model_dir, acc, name="acc_translate_train.json")

# evaluation translate
_, acc_svm = evaluate.svm(Z_train, y_train, Z_translate_test, y_translate_test)
acc_knn = evaluate.knn(Z_train, y_train, Z_translate_test, y_translate_test, k=5)
acc_svd = evaluate.nearsub(Z_train, y_train, Z_translate_test, y_translate_test, n_comp=10)
acc = {"svm": acc_svm, "knn": acc_knn, "nearsub-svd": acc_svd} 
utils.save_params(model_dir, acc, name="acc_translate_test.json")
