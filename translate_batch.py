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
parser.add_argument('--batch_i', type=int, help='batch index')
parser.add_argument('--stride', type=int, default=1, help='stride for lifting')
args = parser.parse_args()

print(args.model_dir)
params = utils.load_params(args.model_dir)
if args.data_dir is None:
    args.data_dir = args.model_dir
X_train = np.load(os.path.join(args.model_dir, 'features', 'X_train_features.npy'))
y_train = np.load(os.path.join(args.model_dir, 'features', 'X_train_labels.npy'))
X_test = np.load(os.path.join(args.model_dir, 'features', 'X_test_features.npy'))
y_test = np.load(os.path.join(args.model_dir, 'features', 'X_test_labels.npy'))
if args.model == 'mnist2d':
    X_translate_train, y_translate_train = F.translate2d(X_train, y_train, stride=7)
    X_translate_test, y_translate_test = F.translate2d(X_test, y_test, stride=7)
elif args.model == 'mnist1d':
    X_translate_train, y_translate_train = F.translate1d(X_train, y_train, stride=10)
    X_translate_test, y_translate_test = F.translate1d(X_test, y_test, stride=10)
print('translate_train', X_translate_train.shape, y_translate_train.shape)
print('translate_test', X_translate_test.shape, y_translate_test.shape)

i = args.batch_i
bs = args.bs
X_translate_train, y_translate_train = X_translate_train[i*bs:(i+1)*bs], y_translate_train[i*bs:(i+1)*bs]
X_translate_test, y_translate_test = X_translate_test[i*bs:(i+1)*bs], y_translate_test[i*bs:(i+1)*bs]
np.save(os.path.join(args.model_dir, 'features', f'y_translate_test_all_b{i}.npy'), y_translate_test)
np.save(os.path.join(args.model_dir, 'features', f'y_translate_train_all_b{i}.npy'), y_translate_train)

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
    layers = [Lift1D(kernel)] + [Fourier1D(l, eta=params['eta'], eps=params['eps'])]
elif args.model == 'mnist' or args.model == 'cifar10_vector':
    layers = [Vector(l, eta=params['eta'], eps=params['eps'])]
else:
    raise NameError(f'model name {args.model} not valid.')
model = Architecture(layers, args.model_dir, len(params['classes']))

print(X_translate_test.shape, X_translate_train.shape)
# forward pass 
Z_translate_train = model(X_translate_train).real
np.save(os.path.join(args.model_dir, 'features', f'X_translate_train_all_b{i}.npy'), X_translate_train)
np.save(os.path.join(args.model_dir, 'features', f'Z_translate_train_all_b{i}.npy'), Z_translate_train)

Z_translate_test = model(X_translate_test).real
np.save(os.path.join(args.model_dir, 'features', f'X_translate_test_all_b{i}.npy'), X_translate_test)
np.save(os.path.join(args.model_dir, 'features', f'Z_translate_test_all_b{i}.npy'), Z_translate_test)


# plot and evaluate
#plot.plot_heatmap(Z_translate_test, y_translate_test, 'Z_translate_test_all', args.model_dir)
#plot.plot_heatmap(X_translate_test, y_translate_test, 'X_translate_test_all', args.model_dir)

#_, acc_svm = evaluate.svm(Z_train, y_train, Z_test, y_test)
#acc_knn = evaluate.knn(Z_train, y_train, Z_test, y_test, k=5)
#acc_svd = evaluate.nearsub(Z_train, y_train, Z_test, y_test, n_comp=10)
#acc = {"svm": acc_svm, "knn": acc_knn, "nearsub-svd": acc_svd}
#utils.save_params(args.model_dir, acc, name="acc_test_all.json")

#_, acc_svm = evaluate.svm(Z_train, y_train, Z_translate_test, y_translate_test)
#acc_knn = evaluate.knn(Z_train, y_train, Z_translate_test, y_translate_test, k=5)
#acc_svd = evaluate.nearsub(Z_train, y_train, Z_translate_test, y_translate_test, n_comp=10)
#acc = {"svm": acc_svm, "knn": acc_knn, "nearsub-svd": acc_svd}
#utils.save_params(args.model_dir, acc, name="acc_translate_test_all.json")
