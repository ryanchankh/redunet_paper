import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from redunet import Architecture, Vector
import dataset
import evaluate
import plot
import utils


# hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=int, required=True, help='choice of distributions for data')
parser.add_argument('--noise', type=float, default=0.1, help='noise')
parser.add_argument('--samples', type=int, default=100, help="number of samples for initialization")
parser.add_argument('--layers', type=int, default=50, help="number of layers")
parser.add_argument('--eta', type=float, default=0.5, help='learning rate')
parser.add_argument('--eps', type=float, default=0.1, help='eps squared')
parser.add_argument('--tail', type=str, default='',
                    help='extra information to add to folder name')
parser.add_argument('--save_dir', type=str, default='./saved_models/',
                    help='base directory for saving PyTorch model. (default: ./saved_models/)')
args = parser.parse_args()

# pipeline setup
model_dir = os.path.join(args.save_dir, 
						 "gaussian2d",
						 f"data{args.data}_noise{args.noise}",
                         f"samples{args.samples}"
                         f"_layers{args.layers}"
                         f"_eps{args.eps}"
                         f"_eta{args.eta}"
                         f"{args.tail}")
os.makedirs(model_dir, exist_ok=True)
utils.save_params(model_dir, vars(args))
print(model_dir)

# data setup
X_train, y_train, num_classes = dataset.generate_2d(args.data, args.noise, args.samples, shuffle=True)
X_test, y_test, _ = dataset.generate_2d(args.data, args.noise, args.samples, shuffle=False)

# model setup
layers = [Vector(args.layers, eta=args.eta, eps=args.eps)]
model = Architecture(layers, model_dir, num_classes)

# train/test pass
print("Forward pass - train features")
Z_train = model(X_train, y_train)
utils.save_loss(model.loss_dict, model_dir, "train")
print("Forward pass - test features")
Z_test = model(X_test)
utils.save_loss(model.loss_dict, model_dir, "test")

# save features
utils.save_features(model_dir, "X_train", X_train, y_train)
utils.save_features(model_dir, "X_test", X_test, y_test)
utils.save_features(model_dir, "Z_train", Z_train, y_train)
utils.save_features(model_dir, "Z_test", Z_test, y_test)

# evaluation train
_, acc_svm = evaluate.svm(Z_train, y_train, Z_train, y_train)
acc_knn = evaluate.knn(Z_train, y_train, Z_train, y_train, k=5)
acc_svd = evaluate.nearsub(Z_train, y_train, Z_train, y_train, n_comp=1)
acc = {"svm": acc_svm, "knn": acc_knn, "nearsub-svd": acc_svd} 
utils.save_params(model_dir, acc, name="acc_train.json")

# evaluation test
_, acc_svm = evaluate.svm(Z_train, y_train, Z_test, y_test)
acc_knn = evaluate.knn(Z_train, y_train, Z_test, y_test, k=5)
acc_svd = evaluate.nearsub(Z_train, y_train, Z_test, y_test, n_comp=1)
acc = {"svm": acc_svm, "knn": acc_knn, "nearsub-svd": acc_svd} 
utils.save_params(model_dir, acc, name="acc_test.json")

# plot
plot.plot_combined_loss(model_dir)
plot.plot_heatmap(X_train, y_train, "X_train", model_dir)
plot.plot_heatmap(X_test, y_test, "X_test", model_dir)
plot.plot_heatmap(Z_train, y_train, "Z_train", model_dir)
plot.plot_heatmap(Z_test, y_test, "Z_test", model_dir)
plot.plot_2d(X_train, y_train, "X_train", model_dir)
plot.plot_2d(X_test, y_test, "X_test", model_dir)
plot.plot_2d(Z_train, y_train, "Z_train", model_dir)
plot.plot_2d(Z_test, y_test, "Z_test", model_dir)
