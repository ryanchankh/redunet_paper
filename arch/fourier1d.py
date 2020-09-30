import time
import numpy as np

from .vector import Vector
import train_func as tf


class Fourier1D(Vector):
    def __init__(self, layers, eta, eps, lmbda=5000):
        super().__init__(layers, eta, eps, lmbda)

    def forward(self, layer, V, y=None, init=False):
        if init:
            self.init(V, y)
            self.save_weights(layer)
        else:
            self.load_weights(layer)
        expd = np.einsum("bi...,ih...->bh...", V, self.E.conj(), optimize=True)
        comp = np.stack([np.einsum("bi...,ih...->bh...", V, C_j.conj(), optimize=True) \
                for C_j in self.Cs])
        clus, y_approx = self.nonlinear(comp)
        V = V + self.eta * (expd - clus)
        V = tf.normalize(V)
        return V, y_approx

    def compute_E(self, V):
        m, C, T = V.shape
        alpha = C / (m * self.eps)
        pre_inv = alpha * tf.batch_cov(V, self.arch.batch_size) \
                  + np.eye(C)[..., np.newaxis]
        E = np.empty_like(pre_inv)
        for t in range(T):
            E[:, :, t] = alpha * np.linalg.inv(pre_inv[:, :, t])
        self.E = E

    def compute_Cs(self, V, y):
        m, C, T = V.shape
        Cs = np.empty((self.num_classes, C, C, T), dtype=np.complex)
        for j in np.arange(self.num_classes):
            V_j = V[y==j]
            m_j = V_j.shape[0]
            if m_j == 0:
                continue
            alpha_j = C / (m_j * self.eps)
            pre_inv = alpha_j * tf.batch_cov(V_j, self.arch.batch_size) \
                + np.eye(C)[..., np.newaxis]
            for t in range(T):
                Cs[j, :, :, t] =  alpha_j * np.linalg.inv(pre_inv[:, :, t])
        self.Cs = Cs

    def compute_loss(self, V, y):
        m, C, T = V.shape
        alpha = C / (m * self.eps)
        cov = alpha * tf.batch_cov(V, self.arch.batch_size) \
                + np.eye(C)[..., np.newaxis]
        loss_expd = np.sum([np.linalg.slogdet(cov[:, :, t])[1] for t in range(T)])  / (2 * T)

        loss_comp = 0.
        Cs = np.empty((self.num_classes, C, C, T), dtype=np.complex)
        for j in range(self.num_classes):
            V_j = V[y==int(j)]
            m_j = V_j.shape[0]
            if m_j == 0:
                continue
            alpha_j = C / (m_j * self.eps) 
            cov_j = alpha_j * tf.batch_cov(V_j, self.arch.batch_size) \
                        + np.eye(C)[..., np.newaxis]
            loss_comp += m_j / m * np.sum([np.linalg.slogdet(cov_j[:, :, t])[1] for t in range(T)])  / (2 * T)
        return loss_expd - loss_comp, loss_expd, loss_comp

    def preprocess(self, X):
        Z = tf.normalize(X)
        return np.fft.fft(X, norm='ortho', axis=2)

    def postprocess(self, X):
        Z = np.fft.ifft(X, norm='ortho', axis=2)
        return tf.normalize(Z)

