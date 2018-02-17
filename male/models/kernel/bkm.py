from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import time
import sys
import numpy as np
import tensorflow as tf

from male.utils.generic_utils import Progbar

from .bkm_base import BaseBKM


class BKM(BaseBKM):
    """ Using Stein Variational Gradient Descent Framework to learn model
    """

    def __init__(self,
                 model_name="BKM",
                 **kwargs):
        super(BKM, self).__init__(model_name=model_name, **kwargs)

    def _build_model(self, x):
        self.input_dim = x.shape[1]

        scale_w_value = 1.0
        if self.scale_w:
            scale_w_value = 1.0 / self.w_regular
        if self.linear:
            self.w_init = scale_w_value * np.random.normal(0, 1, [self.num_samples_params * self.num_classes, self.input_dim])
            self.w_init = self.w_init.reshape(self.num_samples_params, self.num_classes * self.input_dim)

            self.params_init = self.w_init
            self.params = []
            self.w_lst = []
            for ip in range(self.num_samples_params):
                param = tf.Variable(self.w_init[ip, :], dtype=tf.float32)
                w = tf.reshape(param, [self.num_classes, self.input_dim])
                self.params.append(param)
                self.w_lst.append(w)
            # self.params = tf.Variable(self.w_init, dtype=tf.float32)
            # self.w = self.params
            # self.w_lst = tf.sparse_split(
            #     sp_input=tf.sparse_reshape(self.w, [self.num_samples_params, self.num_classes, self.input_dim]),
            #     num_split=self.num_samples_params, axis=0)
        else:
            self.w_init = scale_w_value * np.random.normal(0, 1, [self.num_samples_params * self.num_classes, self.rf_2dim_pad])
            self.w_init = self.w_init.reshape(self.num_samples_params, self.num_classes * self.rf_2dim_pad)
            self.sigma_init = np.random.normal(0, 1, [self.num_samples_params, self.input_dim]) + self.x_kernel_width

            self.params_init = np.concatenate([self.w_init, self.sigma_init], axis=1)
            self.params = tf.Variable(self.params_init, dtype=tf.float32)

            self.w = self.params[:, :self.num_classes*self.rf_2dim_pad]
            self.w_lst = tf.unstack(
                tf.reshape(self.w, [self.num_samples_params, self.num_classes, self.rf_2dim_pad]), axis=0)

            self.sigma = self.params[:, -self.input_dim:]
            self.sigma_lst = tf.unstack(self.sigma, axis=0)

        super(BKM, self)._build_model(x)

        if self.linear:
            self.grad_loss_all = self.obj_grad_w_all_reshape
        else:
            self.grad_loss_all = tf.concat([self.obj_grad_w_all_reshape, self.loss_grad_sigma_all], axis=1)

        self.params_update, self.kxy_params, self.dxkxy_params = self._create_update_vars_op(
            self.grad_loss_all, self.params, self.step_size)

        self.init_variables = tf.global_variables_initializer()
        self.variables = tf.trainable_variables()

    def _fit_loop(self, x, y,
                  do_validation=False,
                  x_valid=None, y_valid=None,
                  callbacks=None, callback_metrics=None):
        self.x_ = x
        self.y_ = y

        if self.info == 1:
            progbar = Progbar(self.num_iters, show_steps=1)

        num_samples = x.shape[0]
        self.input_dim = x.shape[1]

        if not self.linear:
            self.eps_omega_const = np.random.randn(self.rf_dim, self.input_dim)

        self.params_value = self.params_init

        for it in range(self.num_iters):
            epoch_logs = {}
            callbacks.on_epoch_begin(self.epoch)
            if it > 20:
                if ((it + 1) % self.freq_calc_metrics) == 0:
                    if self.info == 1:
                        progbar.update(it)
                    elif self.info > 0:
                        print('Iter', it,
                              '| Remaining: %.0fs' % ((time.time() - self.start_time) / it * (self.num_iters - it)))
                    if 'train_loss' in callback_metrics:
                        train_mean_loss, _ = self.update_forward(x, y)
                        epoch_logs['train_loss'] = train_mean_loss
                    if 'valid_loss' in callback_metrics:
                        valid_mean_loss, _ = self.update_forward(x_valid, self._transform_labels(y_valid))
                        epoch_logs['valid_loss'] = valid_mean_loss

            idx_samples = np.random.permutation(num_samples)[:self.batch_size]

            self.params_value = self._update_vars(
                it, self.params_update, self.kxy_params, self.dxkxy_params, self.params_value,
                self.params_kernel_width, idx_samples)

            if (self.info > 1) and not self.linear:
                print(np.average(self.params_value[:, -self.input_dim:], axis=0))

            if it > 20:
                if ((it + 1) % self.freq_calc_metrics) == 0:
                    # if (((it + 1) / self.freq_calc_metrics) + 1) * self.freq_calc_metrics > self.num_iters:
                    #     print('Finish drawing')
                    self.epoch += 1
                    callbacks.on_epoch_end(self.epoch, epoch_logs)

        for callback in self.callbacks:
            if 'fig' in callback.__dict__:
                try:
                    callback.fig.savefig('plot_' + '_'.join(sys.argv[1:]) + '.png')
                except:
                    print("Cannot save fig")
