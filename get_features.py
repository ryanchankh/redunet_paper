import os
import numpy as np

from arch import (
    Architecture,
    Lift1D,
    Fourier1D
)
import dataset
import train_func as tf
import utils
import plot



model_dir = "./saved_models/mnist_rotation-classes01/samples1000_layers3500_eps0.1_eta0.5/"
m = 500  # samples each class
bs = 20 # batch size
stride = 5

params = utils.load_params(model_dir)
classes = np.array(params['classes'])

X_train = np.load(os.path.join(model_dir, "features", "X_train_features.npy"))
y_train = np.load(os.path.join(model_dir, "features", "X_train_labels.npy"))
X_test = np.load(os.path.join(model_dir, "features", "X_test_features.npy"))
y_test = np.load(os.path.join(model_dir, "features", "X_test_labels.npy"))

layers = [Fourier1D(200, eta=params['eta'], eps=params['eps'], lmbda=params['lmbda'])]
model = Architecture(layers, model_dir, classes.size)

model.blocks[0].num_classes = model.num_classes 
model.blocks[0].compute_gam(y_train)

X_test_filter, y_test_filter = tf.filter_class(X_test, y_test, classes, m)
for batch_idx in range(0, m * classes.size, bs):
    X_batch = X_test_filter[batch_idx:batch_idx+bs]
    y_batch = y_test_filter[batch_idx:batch_idx+bs]

    X_translate, y_translate = tf.translate_all(X_batch, y_batch, stride=stride)
    Z_translate = model(X_translate).real

    np.save(os.path.join(model_dir, "features", f"X_translate_features-{batch_idx}-{batch_idx+bs}"), X_translate)
    np.save(os.path.join(model_dir, "features", f"Z_translate_features-{batch_idx}-{batch_idx+bs}"), Z_translate)
    np.save(os.path.join(model_dir, "features", f"y_translate_labels-{batch_idx}-{batch_idx+bs}"), y_translate)
