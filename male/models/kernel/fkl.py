from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import scipy.special

np.seterr(all='raise')

from scipy.optimize import check_grad

from . import FOGD
from ...utils.generic_utils import make_batches

INF = 1e+8


class FKL(FOGD):
    """Reparameterized Random Features
    """

    def __init__(self,
                 model_name="FKL",
                 mode='batch',  # {'batch', 'online'}
                 learning_rate_gamma=0.005,
                 e=None,
                 b=None,
                 **kwargs):
        super(FKL, self).__init__(model_name=model_name, **kwargs)
        self.mode = mode
        self.learning_rate_gamma = learning_rate_gamma
        self.e = e
        self.b = b

    def _init(self):
        super(FKL, self)._init()
        self.gamma_ = None
        self.num_features = 0  # number of data features
        if self.learning_rate_gamma < 0:
            self.learning_rate_gamma = self.learning_rate

    def _init_params(self, x):
        super(FKL, self)._init_params(x)
        if self.num_classes > 2:
            self.w = 0.01 * self.random_engine.randn(self.D, self.num_classes)
        else:
            self.w = 0.01 * self.random_engine.randn(self.D)

        self.num_features = x.shape[1]
        self.gamma_ = np.log(self.gamma) * np.ones(self.num_features)  # Nx1

        if self.e is None:
            self.e = np.random.uniform(0, 1, (self.num_features, self.D))
            self.e = np.sqrt(2) * scipy.special.erfinv(2 * self.e - 1)
            self.b = np.random.uniform(0, 2 * np.pi, self.D)

    def _get_wxy(self, wx, y):
        m = len(y)  # batch size
        idx = range(m)
        mask = np.ones([m, self.num_classes], np.bool)
        mask[idx, y] = False
        z = np.argmax(wx[mask].reshape([m, self.num_classes - 1]), axis=1)
        z += (z >= y)
        return wx[idx, y] - wx[idx, z], z

    def get_gamma_grad(self, x, phi, sinwx, dphi, *args, **kwargs):
        gamma = kwargs['gamma'] if 'gamma' in kwargs else self.gamma_  # (N,)

        m = x.shape[0]  # batch size
        # gradient of \phi w.r.t \omega
        dpo = np.zeros([m, self.D, self.num_features])  # (M,2D,N)

        # broadcasting
        # drw[:, :self.D, :] = -X[:, np.newaxis, :] * sinwx[:, :, np.newaxis] * self.eps_.T[np.newaxis, :, :]  # (M,D,N)
        # drw[:, self.D:, :] = X[:, np.newaxis, :] * coswx[:, :, np.newaxis] * self.eps_.T[np.newaxis, :, :]  # (M,D,N)
        # dlg = drw.reshape([M * 2 * self.D, self.N_]).T.dot(dr.reshape(M * 2 * self.D)) * np.exp(g)

        # einsum
        dpo[:, :, :] = np.einsum("mn,md,nd->mdn", -x, sinwx, self.e)  # (M,D,N)
        dlg = np.einsum("mdn,md->n", dpo, dphi) * np.exp(gamma)

        return dlg

    def get_grad(self, x, y, *args, **kwargs):
        m = x.shape[0]  # batch size
        w = kwargs['w'] if 'w' in kwargs else self.w  # (D,C)
        gamma = kwargs['gamma'] if 'gamma' in kwargs else self.gamma_  # (N,)
        if 'phi' in kwargs:
            phi = kwargs['phi']
            sinwx = kwargs['sinwx']
        else:
            phi, sinwx = self._get_phi(x, **kwargs)  # (M,D)

        wx = kwargs['wx'] if 'wx' in kwargs else phi.dot(w)  # (M,C)

        dw = self.lbd * w  # (D,C)
        dgamma = np.zeros(gamma.shape)  # (N,)
        if self.num_classes > 2:
            wxy, z = self._get_wxy(wx, y)
            if self.loss == 'hinge':
                d = (wxy[:, np.newaxis] < 1) * phi  # (M,D)
                dphi = -w[:, y[wxy < 1]].T + w[:, z[wxy < 1]].T  # (M,D)
                dgamma += self.get_gamma_grad(x[wxy < 1], phi[wxy < 1], sinwx[wxy < 1], dphi, gamma=gamma) / m
            else:  # logit loss
                c = np.exp(-wxy - np.logaddexp(0, -wxy))[:, np.newaxis]
                d = c * phi
                dphi = -c * (w[:, y].T - w[:, z].T)  # (M,D)
                dgamma += self.get_gamma_grad(x, phi, sinwx, dphi, gamma=gamma) / m
            for i in range(self.num_classes):
                dw[:, i] += -d[y == i].sum(axis=0) / m
                dw[:, i] += d[z == i].sum(axis=0) / m
        else:
            if self.loss == 'hinge':
                wxy = y * wx
                dw += np.sum(-y[wxy < 1, np.newaxis] * phi[wxy < 1], axis=0) / m
                dphi = -y[wxy < 1, np.newaxis] * w  # (M,D)
                dgamma += self.get_gamma_grad(x[wxy < 1], phi[wxy < 1], sinwx[wxy < 1], dphi, gamma=gamma) / m
            elif self.loss == 'l1':
                wxy = np.sign(wx - y)[:, np.newaxis]
                dw += (wxy * phi).mean(axis=0)
                dphi = wxy * w  # (M,D)
                dgamma = self.get_gamma_grad(x, phi, sinwx, dphi, gamma=gamma) / m
            elif self.loss == 'l2':
                wxy = (wx - y)[:, np.newaxis]
                dw += (wxy * phi).mean(axis=0)
                dphi = wxy * w  # (M,D)
                dgamma = self.get_gamma_grad(x, phi, sinwx, dphi, gamma=gamma) / m
            elif self.loss == 'logit':
                wxy = y * wx
                c = (-y * np.exp(-wxy - np.logaddexp(0, -wxy)))[:, np.newaxis]
                dw += np.mean(c * phi, axis=0)
                dphi = c * w  # (M,D)
                dgamma += self.get_gamma_grad(x, phi, sinwx, dphi, gamma=gamma) / m
            elif self.loss == 'eps_insensitive':
                wxy = np.abs(y - wx) > self.eps
                c = np.sign(wx - y)[:, np.newaxis]
                d = c * phi
                dw += d[wxy].sum(axis=0) / m
                dphi = c[wxy] * w  # (M,D)
                dgamma += self.get_gamma_grad(x[wxy], phi[wxy], sinwx[wxy], dphi, gamma=gamma) / m
        return dw, dgamma

    def _fit_loop(self, x, y,
                  do_validation=False,
                  x_valid=None, y_valid=None,
                  callbacks=None, callback_metrics=None):

        if self.mode == 'online':  # online setting
            y0 = self._decode_labels(y)
            if self.avg_weight:
                w_avg = np.zeros(self.w_.shape)

            mistake = 0.0
            for t in range(x.shape[0]):
                phi, _ = self._get_phi(x[[t]])

                wx = phi.dot(self.w)  # (x,)
                if self.task == 'classification':
                    if self.num_classes == 2:
                        y_pred = self._decode_labels(np.uint8(wx >= 0))[0]
                    else:
                        y_pred = self._decode_labels(np.argmax(wx))
                    mistake += (y_pred != y0[t])
                else:
                    mistake += (wx[0] - y0[t]) ** 2
                dw, dgamma = self.get_grad(x[[t]], y[[t]], phi=phi, wx=wx)  # compute gradients

                # update parameters
                self.w -= self.learning_rate * dw
                self.gamma_ -= self.learning_rate_gamma * dgamma

                # update the average of parameters
                if self.avg_weight:
                    w_avg += self.w

            if self.avg_weight:
                self.w = w_avg / x.shape[0]

            self.mistake = mistake / x.shape[0]

        else:  # batch setting
            batches = make_batches(x.shape[0], self.batch_size)
            while (self.epoch < self.num_epochs) and (not self.stop_training):
                epoch_logs = {}
                callbacks.on_epoch_begin(self.epoch)
                self.num_samples = x.shape[0]

                for batch_idx, (batch_start, batch_end) in enumerate(batches):
                    batch_logs = {'batch': batch_idx,
                                  'size': batch_end - batch_start}
                    callbacks.on_batch_begin(batch_idx, batch_logs)

                    idx_samples = np.random.permutation(self.num_samples)[:self.batch_size]

                    x_batch = x[idx_samples, :]
                    y_batch = y[idx_samples]

                    dw, dgamma = self.get_grad(x_batch, y_batch)

                    self.w -= self.learning_rate * dw
                    self.gamma_ -= self.learning_rate_gamma * dgamma

                    batch_logs.update(self._on_batch_end(x_batch, y_batch))
                    callbacks.on_batch_end(batch_idx, batch_logs)

                if do_validation:
                    outs = self._on_batch_end(x_valid, self._transform_labels(y_valid))
                    for key, value in outs.items():
                        epoch_logs['val_' + key] = value

                callbacks.on_epoch_end(self.epoch, epoch_logs)
                self._on_epoch_end()

    def predict(self, x):
        if x.ndim < 2:
            x = x.copy()[..., np.newaxis]

        y = np.ones(x.shape[0])
        batches = make_batches(x.shape[0], self.batch_size)
        for batch_idx, (batch_start, batch_end) in enumerate(batches):
            x_batch = x[batch_start:batch_end]
            phi, _ = self._get_phi(x_batch)
            wx = phi.dot(self.w)
            if self.task == 'classification':
                if self.num_classes == 2:
                    y[batch_start:batch_end] = self._decode_labels(np.uint8(wx >= 0))
                else:
                    y[batch_start:batch_end] = self._decode_labels(np.argmax(wx, axis=1))
            else:
                y[batch_start:batch_end] = wx
        return y

    def _predict(self, x):
        if x.ndim < 2:
            x = x.copy()[..., np.newaxis]

        y = np.ones(x.shape[0])
        batches = make_batches(x.shape[0], self.batch_size)
        for batch_idx, (batch_start, batch_end) in enumerate(batches):
            x_batch = x[batch_start:batch_end]
            phi, _ = self._get_phi(x_batch)
            wx = phi.dot(self.w)
            if self.task == 'classification':
                if self.num_classes == 2:
                    y[batch_start:batch_end] = np.uint8(wx >= 0)
                else:
                    y[batch_start:batch_end] = np.argmax(wx, axis=1)
            else:
                y[batch_start:batch_end] = wx
        return y

    def _get_phi(self, x, **kwargs):
        gamma = kwargs['gamma'] if 'gamma' in kwargs else self.gamma_
        omega = np.exp(gamma)[:, np.newaxis] * self.e  # NxD

        # phi = np.zeros([x.shape[0], self.D])  # MxD
        xo = x.dot(omega)
        phi = np.cos(xo + self.b)
        sin_xo = np.sin(xo + self.b)
        return phi, sin_xo

    def _roll_params(self):
        return np.concatenate([super(FKL, self)._roll_params(),
                               np.ravel(self.gamma_.copy())])

    def _unroll_params(self, w):
        ww = super(FKL, self)._unroll_params(w)
        ww = tuple([ww]) if not isinstance(ww, tuple) else ww
        idx = np.sum([i.size for i in ww], dtype=np.int)
        gamma = w[idx:idx + self.gamma_.size].reshape(self.gamma_.shape).copy()
        return ww + (gamma,)

    def get_loss(self, x, y, *args, **kwargs):
        if 'gamma' not in kwargs:
            kwargs['gamma'] = self.gamma_

        w = kwargs['w'] if 'w' in kwargs else self.w
        phi, _ = self._get_phi(x, **kwargs)
        wx = phi.dot(w)

        f = (self.lbd / 2) * np.sum(w * w)
        if self.num_classes > 2:
            wxy, z = self._get_wxy(wx, y)
            if self.loss == 'hinge':
                f += np.maximum(0, 1 - wxy).mean()
            else:  # logit loss
                f += np.logaddexp(0, -wxy).mean()
        else:
            if self.loss == 'hinge':
                f += np.maximum(0, 1 - y * wx).mean()
            elif self.loss == 'l1':
                f += np.abs(y - wx).mean()
            elif self.loss == 'l2':
                f += np.mean(0.5 * ((y - wx) ** 2))
            elif self.loss == 'logit':
                f += np.logaddexp(0, -y * wx).mean()
            elif self.loss == 'eps_insensitive':
                f += np.maximum(0, np.abs(y - wx) - self.eps).mean()

        return f

    def _get_loss_check_grad(self, w, x, y):
        ww, gamma = self._unroll_params(w)
        return self.get_loss(x, y, w=ww, gamma=gamma)

    def _get_grad_check_grad(self, w, x, y):
        ww, gamma = self._unroll_params(w)
        dw, dgamma = self.get_grad(x, y, w=ww, gamma=gamma)
        return np.concatenate([np.ravel(dw), np.ravel(dgamma)])

    def check_grad_online(self, x, y):
        """Check gradients of the model using data X and label y if available
         """
        self._init()

        if y is not None:
            # encode labels
            y = self._encode_labels(y)

        # initialize weights
        self._init_params(x)

        print("Checking gradient... ", end='')

        s = 0.0
        for t in range(x.shape[0]):
            s += check_grad(self._get_loss_check_grad,
                            self._get_grad_check_grad,
                            self._roll_params(),
                            x[[t]], y[[t]])

            dw, dgamma = self.get_grad(x[[t]], y[[t]])
            self.w -= self.learning_rate * dw
            self.gamma_ -= self.learning_rate_gamma * dgamma

        s /= x.shape[0]
        print("diff = %.8f" % s)
        return s

    def score(self, x, y, sample_weight=None):
        if self.exception:
            return -INF
        if self.mode == 'online':
            return -self.mistake
        else:
            return super(FOGD, self).score(x, y)

    def get_params(self, deep=True):
        out = super(FKL, self).get_params(deep=deep)
        param_names = FKL._get_param_names()
        out.update(self._get_params(param_names=param_names, deep=deep))
        return out
