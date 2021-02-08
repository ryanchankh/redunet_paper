import argparse
import os
import numpy as np

from redunet import (
    Architecture,
    Lift1D,
    Lift2D,
    Fourier1D,
    Fourier2D,
    Vector
)
import dataset
import evaluate
from torchvision.datasets import MNIST
import functionals as F
import utils
import plot


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, required=True, help='model directory')
parser.add_argument('--data_dir', type=str, default=None, help='data directory')
parser.add_argument('--model', type=str, required=True, help='name model trained')
parser.add_argument('--n', type=int, default=4, help='number of shifts for each dimension')
parser.add_argument('--layers', type=int, default=None, help='number of layers')
parser.add_argument('--bs', type=int, default=100, help='batch size')
parser.add_argument('--stride', type=int, default=1, help='stride for lifting')
parser.add_argument('--batch_indices', nargs="+", type=int, default=None, help="Classes to Keep (Example: 0 1)")
args = parser.parse_args()

print(args.model_dir)
params = utils.load_params(args.model_dir)
if args.data_dir is None:
    args.data_dir = args.model_dir
testset = MNIST("./data/mnist/", train=False, download=True)
X_train = np.load(os.path.join(args.model_dir, 'features', 'X_train_features.npy'))
y_train = np.load(os.path.join(args.model_dir, 'features', 'X_train_labels.npy'))
Z_train = np.load(os.path.join(args.model_dir, 'features', 'Z_train_features.npy'))
X_test, y_test = testset.data, testset.targets
X_test, y_test = F.get_n_each(X_test, y_test, 10)
if args.model == 'mnist2d':
    X_test = np.expand_dims(X_test, 1)
    X_translate_test, y_translate_test = F.translate2d(X_test, y_test, stride=7)
elif args.model == 'mnist1d':
    X_test = F.convert2polar(X_test, params['channels'], params['time'])
    X_translate_test, y_translate_test = F.translate1d(X_test, y_test, stride=10)
print(X_translate_test.shape, y_translate_test.shape)
print(X_test.shape, y_test.shape)

# save features
np.save(os.path.join(args.model_dir, 'features', 'X_translate_test_all.npy'), X_translate_test)
np.save(os.path.join(args.model_dir, 'features', 'y_translate_test_all.npy'), y_translate_test)

if args.layers is None:
    l = params['layers']
else:
    l = args.layers

print(args.model)
if args.model == 'mnist2d':
    kernel = np.load(os.path.join(args.model_dir, 'features', 'kernel.npy'))
    layers = [Lift2D(kernel)] + [Fourier2D(l, eta=params['eta'], eps=params['eps'])]
elif args.model == 'mnist1d':
    kernel = np.load(os.path.join(args.model_dir, 'features', 'kernel.npy'))
    print(kernel.shape)
    layers = [Lift1D(kernel)] + [Fourier1D(l, eta=params['eta'], eps=params['eps'])]
elif args.model == 'mnist' or args.model == 'cifar10_vector':
    layers = [Vector(l, eta=params['eta'], eps=params['eps'])]
else:
    raise NameError(f'model name {args.model} not valid.')
model = Architecture(layers, args.model_dir, len(params['classes']))

# forward pass 
Z_translate_test = model(X_translate_test).real
Z_test = model(X_test).real
utils.save_loss(model.loss_dict, args.model_dir, 'test_all')

# normalize
Z_train = F.normalize(Z_train.reshape(Z_train.shape[0], -1))
Z_test = F.normalize(Z_test.reshape(Z_test.shape[0], -1))
X_translate_test = F.normalize(X_translate_test.reshape(X_translate_test.shape[0], -1))
Z_translate_test = F.normalize(Z_translate_test.reshape(Z_translate_test.shape[0], -1))

# save features
np.save(os.path.join(args.model_dir, 'features', 'Z_test_all.npy'), Z_test)
np.save(os.path.join(args.model_dir, 'features', 'Z_translate_test_all.npy'), Z_translate_test)

# plot and evaluate
print(Z_translate_test.shape)
plot.plot_heatmap(Z_translate_test, y_translate_test, 'Z_translate_test_all', args.model_dir)
plot.plot_heatmap(X_translate_test, y_translate_test, 'X_translate_test_all', args.model_dir)

_, acc_svm = evaluate.svm(Z_train, y_train, Z_test, y_test)
acc_knn = evaluate.knn(Z_train, y_train, Z_test, y_test, k=5)
acc_svd = evaluate.nearsub(Z_train, y_train, Z_test, y_test, n_comp=10)
acc = {"svm": acc_svm, "knn": acc_knn, "nearsub-svd": acc_svd}
utils.save_params(args.model_dir, acc, name="acc_test_all.json")

_, acc_svm = evaluate.svm(Z_train, y_train, Z_translate_test, y_translate_test)
acc_knn = evaluate.knn(Z_train, y_train, Z_translate_test, y_translate_test, k=5)
acc_svd = evaluate.nearsub(Z_train, y_train, Z_translate_test, y_translate_test, n_comp=10)
acc = {"svm": acc_svm, "knn": acc_knn, "nearsub-svd": acc_svd}
utils.save_params(args.model_dir, acc, name="acc_translate_test_all.json")
