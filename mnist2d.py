import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST


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
parser.add_argument('--layers', type=int, help="number of layers")
parser.add_argument('--eta', type=float, help='learning rate')
parser.add_argument('--eps', type=float, help='eps squared')
parser.add_argument('--lmbda', type=float, default=5000, help='lambda')
parser.add_argument('--tail', type=str, default='',
                    help='extra information to add to folder name')
parser.add_argument('--save_dir', type=str, default='./saved_models/',
                    help='base directory for saving PyTorch model. (default: ./saved_models/)')
args = parser.parse_args()

model_dir = os.path.join(args.save_dir, f"mnist2d-classes{''.join(map(str, args.classes))}",
                         "samples{}_layers{}_eps{}_eta{}{}"
                         "".format(args.samples, args.layers, args.eps, args.eta, args.tail))
os.makedirs(model_dir, exist_ok=True)
utils.save_params(model_dir, vars(args))

# data loading
# X_train = np.load('./data/mnist_rotation/mnist_X_train_C5_T200.npy')
# y_train = np.load('./data/mnist_rotation/mnist_y_train_C5_T200.npy')
# X_test = np.load('./data/mnist_rotation/mnist_X_test_C5_T200.npy')
trainset = MNIST("./data/mnist/", train=True, download=True)
X_train, y_train = trainset.data, trainset.targets
X_train = np.expand_dims(X_train, 1)
testset = MNIST("./data/mnist/", train=False, download=True)
X_test, y_test = testset.data, testset.targets
X_test = np.expand_dims(X_test, 1)
X_train, y_train = F.filter_class(X_train, y_train, args.classes, args.samples)
X_train, y_train = F.shuffle(X_train, y_train)
X_test, y_test = F.filter_class(X_test, y_test, args.classes, args.samples)
X_each, y_each = F.get_n_each(X_test, y_test, 5)
X_translate, y_translate = F.translate2d(X_each, y_each, stride=5)

# setup architecture
kernels = np.random.normal(0, 1, size=(5, 1, 3, 3))
#layers = [Lift2D(kernels)] + [Fourier2D(args.layers, eta=args.eta, eps=args.eps, lmbda=args.lmbda)]
layers = [Fourier2D(args.layers, eta=args.eta, eps=args.eps, lmbda=args.lmbda)]
model = Architecture(layers, model_dir, len(args.classes))

# train/test pass
print("Forward pass - train features")
Z_train = model(X_train, y_train).real
utils.save_loss(model.loss_dict, model_dir, "train")
print("Forward pass - test features")
Z_test = model(X_test).real
utils.save_loss(model.loss_dict, model_dir, "test")
#print("Forward pass - translated features")
#Z_translate = model(X_translate).real
#utils.save_loss(model.loss_dict, model_dir, "translate")

# train and test accuracy
for l in range(args.layers):
    y_approx_train = np.load(f'./labels/train_layer{l}.npy')
    y_approx_test = np.load(f'./labels/test_layer{l}.npy')
    acc_train = evaluate.compute_accuracy(y_approx_train, y_train)
    acc_test = evaluate.compute_accuracy(y_approx_test, y_test)
    print(f'layer: {l} | train: {acc_train:.7f} | test: {acc_test:.7f}')

# save features
utils.save_features(model_dir, "X_train", X_train, y_train)
utils.save_features(model_dir, "X_test", X_test, y_test)
#utils.save_features(model_dir, "X_translate", X_translate, y_translate)
utils.save_features(model_dir, "Z_train", Z_train, y_train)
utils.save_features(model_dir, "Z_test", Z_test, y_test)
#utils.save_features(model_dir, "Z_translate", Z_translate, y_translate)

plt.imsave('./train_img.png', X_train[0, 0])
plt.imsave('./train_img_transformed.png', Z_train[0, 0])
X_train = F.normalize(X_train.reshape(X_train.shape[0], -1))
X_test = F.normalize(X_test.reshape(X_test.shape[0], -1))
#X_translate = F.normalize(X_translate.reshape(X_translate.shape[0], -1))
Z_train = F.normalize(Z_train.reshape(Z_train.shape[0], -1))
Z_test = F.normalize(Z_test.reshape(Z_test.shape[0], -1))
#Z_translate = F.normalize(Z_translate.reshape(Z_translate.shape[0], -1))

# evaluation test
_, acc_svm = evaluate.svm(Z_train, y_train, Z_test, y_test)
acc_knn = evaluate.knn(Z_train, y_train, Z_test, y_test, k=5)
acc_svd = evaluate.nearsub(Z_train, y_train, Z_test, y_test, n_comp=30)
acc = {"svm": acc_svm, "knn": acc_knn, "nearsub-svd": acc_svd} 
utils.save_params(model_dir, acc, name="acc_test.json")

# evaluation translate
#_, acc_svm = evaluate.svm(Z_train, y_train, Z_translate, y_translate)
#acc_knn = evaluate.knn(Z_train, y_train, Z_translate, y_translate, k=5)
#acc_svd = evaluate.nearsub(Z_train, y_train, Z_translate, y_translate, n_comp=30)
#acc = {"svm": acc_svm, "knn": acc_knn, "nearsub-svd": acc_svd} 
#utils.save_params(model_dir, acc, name="acc_translate.json")

# plot
plot.plot_combined_loss(model_dir)
plot.plot_heatmap(X_train, y_train, "X_train", model_dir)
plot.plot_heatmap(X_test, y_test, "X_test", model_dir)
#plot.plot_heatmap(X_translate, y_translate, "X_translate", model_dir)
plot.plot_heatmap(Z_train, y_train, "Z_train", model_dir)
plot.plot_heatmap(Z_test, y_test, "Z_test", model_dir)
#plot.plot_heatmap(Z_translate, y_translate, "Z_translate", model_dir)
