import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10


from redunet import Architecture, Lift2D, Fourier2D
import dataset
import evaluate
import plot
import functionals as F
import utils


# hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--samples', type=int, help="number of samples for initialization")
parser.add_argument('--classes', nargs="+", type=int, help="Classes to Keep (Example: 0 1)")
parser.add_argument('--outchannels', type=int, default =5, help="number of output channels for kernel")
parser.add_argument('--ksize', type=int, default =3, help="kernel size")
parser.add_argument('--layers', type=int, help="number of layers")
parser.add_argument('--eta', type=float, help='learning rate')
parser.add_argument('--eps', type=float, help='eps squared')
parser.add_argument('--lmbda', type=float, default=5000, help='lambda')
parser.add_argument('--tail', type=str, default='',
                    help='extra information to add to folder name')
parser.add_argument('--save_dir', type=str, default='./saved_models/',
                    help='base directory for saving PyTorch model. (default: ./saved_models/)')
args = parser.parse_args()

model_dir = os.path.join(args.save_dir, f"cifar10-classes{''.join(map(str, args.classes))}",
                         "samples{}_layers{}_eps{}_eta{}_outchannels{}_ksize{}{}"
                         "".format(args.samples, args.layers, args.eps, args.eta, args.outchannels, args.ksize, args.tail))
print(model_dir)
os.makedirs(model_dir, exist_ok=True)
utils.save_params(model_dir, vars(args))

# data loading
trainset = CIFAR10("./data/cifar10/", train=True, download=True)
X_train, y_train = trainset.data.transpose(0, 3, 1, 2), np.array(trainset.targets)
testset = CIFAR10("./data/cifar10/", train=False, download=True)
X_test, y_test = testset.data.transpose(0, 3, 1, 2), np.array(testset.targets)
X_train, y_train = F.filter_class(X_train, y_train, args.classes, args.samples)
X_train, y_train = F.shuffle(X_train, y_train)
X_test, y_test = F.filter_class(X_test, y_test, args.classes, args.samples)
X_translate_train, y_translate_train = F.translate2d(*F.get_n_each(X_train, y_train, 2), n=2, stride=4)
X_translate_test, y_translate_test = F.translate2d(*F.get_n_each(X_test, y_test, 2), n=2, stride=4)

# standardize data
mean, std =  0.4733630111949825, 0.25156892869250536
X_train = ((X_train / 255.) - mean) / std
X_test = ((X_test / 255.) - mean) / std

utils.save_features(model_dir, "X_train", X_train, y_train)
utils.save_features(model_dir, "X_test", X_test, y_test)
utils.save_features(model_dir, "X_translate_train", X_translate_train, y_translate_train)
utils.save_features(model_dir, "X_translate_test", X_translate_test, y_translate_test)

# setup architecture
kernels = np.random.normal(0, 1, size=(args.outchannels, 3, args.ksize, args.ksize))
np.save(os.path.join(model_dir, 'features', 'kernels.npy'), kernels)
layers = [Lift2D(kernels)] + [Fourier2D(args.layers, eta=args.eta, eps=args.eps)]
model = Architecture(layers, model_dir, len(args.classes))

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
utils.save_features(model_dir, "Z_train", Z_train, y_train)
utils.save_features(model_dir, "Z_test", Z_test, y_test)
utils.save_features(model_dir, "Z_translate_train", Z_translate_train, y_translate_train)
utils.save_features(model_dir, "Z_translate_test", Z_translate_test, y_translate_test)

X_train = F.normalize(X_train.reshape(X_train.shape[0], -1))
X_test = F.normalize(X_test.reshape(X_test.shape[0], -1))
X_translate_train = F.normalize(X_translate_train.reshape(X_translate_train.shape[0], -1))
X_translate_test = F.normalize(X_translate_test.reshape(X_translate_test.shape[0], -1))
Z_train = F.normalize(Z_train.reshape(Z_train.shape[0], -1))
Z_test = F.normalize(Z_test.reshape(Z_test.shape[0], -1))
Z_translate_train = F.normalize(Z_translate_train.reshape(Z_translate_train.shape[0], -1))
Z_translate_test = F.normalize(Z_translate_test.reshape(Z_translate_test.shape[0], -1))

# evaluation test
_, acc_svm = evaluate.svm(Z_train, y_train, Z_test, y_test)
acc_knn = evaluate.knn(Z_train, y_train, Z_test, y_test, k=5)
acc_svd = evaluate.nearsub(Z_train, y_train, Z_test, y_test, n_comp=10)
acc = {"svm": acc_svm, "knn": acc_knn, "nearsub-svd": acc_svd} 
utils.save_params(model_dir, acc, name="acc_test.json")

# evaluation translate
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
