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
import functionals as F
import utils
import plot


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, required=True, help='model directory')
parser.add_argument('--data_dir', type=str, default=None, help='data directory')
parser.add_argument('--model', type=str, required=True, help='name model trained')
parser.add_argument('--n', type=int, default=4, help='number of shifts for each dimension')
parser.add_argument('--bs', type=int, default=100, help='batch size')
parser.add_argument('--stride', type=int, default=1, help='stride for lifting')
args = parser.parse_args()

print(args.model_dir)
params = utils.load_params(args.model_dir)
if args.data_dir is None:
    args.data_dir = args.model_dir
X_train = np.load(os.path.join(args.data_dir, "features", "X_train_features.npy"))
y_train = np.load(os.path.join(args.data_dir, "features", "X_train_labels.npy"))
X_test = np.load(os.path.join(args.data_dir, "features", "X_test_features.npy"))
y_test = np.load(os.path.join(args.data_dir, "features", "X_test_labels.npy"))
print(X_train.shape, X_test.shape)

if args.model == 'mnist2d':
    kernels = np.random.normal(0, 1, size=(params['outchannels'], 1, params['ksize'], params['ksize']))
    layers = [Lift2D(kernels)] + [Fourier2D(params['layers'], eta=params['eta'], eps=params['eps'])]
elif args.model == 'cifar10':
    kernels = np.random.normal(0, 1, size=(params['outchannels'], 3, params['ksize'], params['ksize']))
    layers = [Lift2D(kernels)] + [Fourier2D(params['layers'], eta=params['eta'], eps=params['eps'])]
elif args.model == 'mnist' or args.model == 'cifar10_vector':
    layers = [Vector(params['layers'], eta=params['eta'], eps=params['eps'])]
else:
    raise NameError(f'model name {args.model} not valid.')
model = Architecture(layers, args.model_dir, len(params['classes']))


for batch_idx in range(0, X_test.shape[0], args.bs):
    print(batch_idx)
    X_batch = X_test[batch_idx:batch_idx+args.bs]
    y_batch = y_test[batch_idx:batch_idx+args.bs]

    X_translate, y_translate = F.translate2d(X_batch, y_batch, n=args.n, stride=args.stride)
    print(X_translate.shape)
    if args.model == 'mnist' or args.model == 'cifar10_vector':
        X_trainslate = X_translate.reshape(X_translate.shape[0], -1)
    Z_translate = model(X_translate).real

    np.save(os.path.join(args.model_dir, "features", f"X_translate_features-{batch_idx}-{batch_idx+args.bs}"), X_translate)
    np.save(os.path.join(args.model_dir, "features", f"Z_translate_features-{batch_idx}-{batch_idx+args.bs}"), Z_translate)
    np.save(os.path.join(args.model_dir, "features", f"y_translate_labels-{batch_idx}-{batch_idx+args.bs}"), y_translate)
