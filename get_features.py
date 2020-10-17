import os
import numpy as np

from redunet import (
    Architecture,
    Lift1D,
    Fourier1D,
    Vector
)
import dataset
import functionals as F
import utils
import plot



model_dir = './saved_models/mnist1d-classes37/time100_channels15_samples1000_layers3500_eps0.1_eta0.5'
m = 500  # samples each class
bs = 50 # batch size
stride = 5

params = utils.load_params(model_dir)
classes = np.array(params['classes'])

X_train = np.load(os.path.join(model_dir, "features", "X_train_features.npy"))
y_train = np.load(os.path.join(model_dir, "features", "X_train_labels.npy"))
X_test = np.load(os.path.join(model_dir, "features", "X_test_features.npy"))
y_test = np.load(os.path.join(model_dir, "features", "X_test_labels.npy"))
print(X_train.shape, X_test.shape)

layers = [Fourier1D(params['layers'], eta=params['eta'], eps=params['eps'], lmbda=params['lmbda'])]
model = Architecture(layers, model_dir, classes.size)

for batch_idx in range(0, m * classes.size, bs):
    X_batch = X_test[batch_idx:batch_idx+bs]
    y_batch = y_test[batch_idx:batch_idx+bs]

    X_translate, y_translate = F.translate1d(X_batch, y_batch, stride=stride)
#    X_translate = F.normalize(X_translate.reshape(X_translate.shape[0], -1))
    Z_translate = model(X_translate).real

    np.save(os.path.join(model_dir, "features", f"X_translate_features-{batch_idx}-{batch_idx+bs}"), X_translate)
    np.save(os.path.join(model_dir, "features", f"Z_translate_features-{batch_idx}-{batch_idx+bs}"), Z_translate)
    np.save(os.path.join(model_dir, "features", f"y_translate_labels-{batch_idx}-{batch_idx+bs}"), y_translate)
