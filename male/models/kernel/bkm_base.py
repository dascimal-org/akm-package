from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import scipy.sparse
import scipy.stats
import numpy as np
import tensorflow as tf

from scipy.spatial.distance import pdist, squareform

from ... import TensorFlowModel
from ...utils.disp_utils import visualize_classification_prediction


class BaseBKM(TensorFlowModel):
    """ Using Stein Variational Gradient Descent Framework to learn model
    """

    def __init__(self,
                 model_name="KernelSteinTF",
                 mode='batch',
                 predict_mode=0,  # 0: avg, 1: voting
                 w_regular=1.0,
                 rf_dim=400,
                 params_kernel_width=-1,
                 x_kernel_width=0.0,
                 step_size=1e-3,
                 num_iters=None,
                 num_samples_params=10,
                 compact_radius=-1.0,
                 sparse=False,
                 linear=False,
                 scale_w=False,
                 opt_name='ada',
                 underflow='raise',
                 freq_calc_metrics=2,
                 info=2,
                 **kwargs):
        super(BaseBKM, self).__init__(model_name=model_name, **kwargs)
        self.mode = mode
        self.predict_mode = predict_mode
        self.w_regular = w_regular
        self.rf_dim = rf_dim
        self.params_kernel_width = params_kernel_width
        self.x_kernel_width = x_kernel_width
        self.step_size = step_size
        self.num_iters = num_iters
        self.num_samples_params = num_samples_params
        self.compact_radius = compact_radius
        self.sparse = sparse
        self.linear = linear
        self.scale_w = scale_w
        self.opt_name = opt_name
        self.underflow = underflow
        self.freq_calc_metrics = freq_calc_metrics
        self.info = info

    def _init(self):
        super(BaseBKM, self)._init()

        self.rf_2dim = self.rf_dim * 2
        self.rf_2dim_pad = self.rf_2dim + 1
        self.rf_scale = 1.0 / np.sqrt(self.rf_dim + 1)
        if self.compact_radius > 0:
            self.rf_2dim_pad += 1

        self.eps_omega_const = None

        self.w_lst = []
        self.sigma_lst = []

        self.cur_param_kernel_width = 0

        if self.underflow != 'raise':
            print('Info: underflow is set to', self.underflow)
            np.seterr(under=self.underflow)

    def _build_model(self, x):
        self.input_dim = x.shape[1]
        if self.sparse:
            self.x = tf.sparse_placeholder(tf.float32, shape=[self.batch_size, self.input_dim])
        else:
            self.x = tf.placeholder(tf.float32, shape=[self.batch_size, self.input_dim])
        self.y = tf.placeholder(tf.int32, shape=[self.batch_size])
        self.current_iter = tf.placeholder(tf.int32)

        num_samples = x.shape[0]
        if self.num_iters is None:
            if num_samples <= 30000:
                self.num_iters = int(0.4 * num_samples)
            elif num_samples <= 400000:
                self.num_iters = int(0.1 * num_samples)
            elif num_samples <= 8000000:
                self.num_iters = int(0.04 * x.shape[0])
            else:
                self.num_iters = int(0.02 * x.shape[0])

        if not self.linear:
            self.eps_omega = tf.placeholder(tf.float32, shape=[self.rf_dim, self.input_dim])

            self.x_rf_bias = tf.constant(np.ones((self.batch_size, 1)), dtype=tf.float32)

            self.omega_lst = []
            self.omega_x_lst = []
            self.x_rf_lst = []
            for sigma in self.sigma_lst:
                omega = self.eps_omega * sigma  # (rf_dim, input_dim)
                if self.sparse:
                    omega_x = tf.sparse_tensor_dense_matmul(self.x, tf.matrix_transpose(omega))
                else:
                    omega_x = tf.matmul(self.x, omega, transpose_b=True)
                x_rf = tf.concat([tf.cos(omega_x), tf.sin(omega_x), self.x_rf_bias], 1)

                if self.compact_radius > 0:
                    compact_ext = tf.constant(np.ones((self.batch_size, 1)) * self.compact_radius, dtype=tf.float32)
                    x_rf = tf.concat([x_rf, compact_ext], axis=1)
                    compact_scale = self.compact_radius / tf.norm(x_rf, axis=1)
                    x_rf *= tf.transpose(
                        tf.reshape(tf.tile(compact_scale, [self.rf_2dim_pad]), [self.rf_2dim_pad, self.batch_size]))

                self.omega_lst.append(omega)
                self.omega_x_lst.append(omega_x)
                self.x_rf_lst.append(x_rf)

        self.wx_lst = []
        self.loss_lst = []
        self.loss_grad_w_lst = []
        self.loss_grad_sigma_lst = []
        self.obj_grad_w_lst = []
        for it in range(self.num_samples_params):
            if self.linear:
                if self.sparse:
                    wx = tf.sparse_tensor_dense_matmul(self.x, tf.matrix_transpose(self.w_lst[it]))
                else:
                    wx = tf.matmul(self.x, tf.matrix_transpose(self.w_lst[it]))
            else:
                wx = tf.matmul(self.x_rf_lst[it], tf.matrix_transpose(self.w_lst[it]))

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=wx, labels=self.y)
            loss_grad_w = tf.reduce_mean(tf.gradients(loss, self.w_lst[it]), axis=0)
            if not self.linear:
                loss_grad_sigma = tf.reduce_mean(tf.gradients(loss, self.sigma_lst[it]), axis=0)

            obj_grad_w = self.w_regular * self.w_lst[it] + loss_grad_w

            self.wx_lst.append(wx)
            self.loss_lst.append(loss)
            self.loss_grad_w_lst.append(loss_grad_w)
            if not self.linear:
                self.loss_grad_sigma_lst.append(loss_grad_sigma)
            self.obj_grad_w_lst.append(obj_grad_w)

        self.loss_grad_w_all = tf.stack(self.loss_grad_w_lst)
        if not self.linear:
            self.loss_grad_sigma_all = tf.stack(self.loss_grad_sigma_lst)
        if self.linear:
            self.obj_grad_w_all_reshape = tf.reshape(
                tf.stack(self.obj_grad_w_lst),
                (self.num_samples_params, self.num_classes * self.input_dim))
        else:
            self.obj_grad_w_all_reshape = tf.reshape(
                tf.stack(self.obj_grad_w_lst),
                (self.num_samples_params, self.num_classes * self.rf_2dim_pad))

        self.wx_all = tf.stack(self.wx_lst)
        self.loss_all = tf.stack(self.loss_lst)

    def _create_update_vars_op(self, grad_func, params, step_size_value, fudge_factor_value=1e-6, alpha_value=0.9):
        if self.linear and self.sparse:
            params_dim = params[0].get_shape().as_list()[0]
        else:
            params_dim = params.get_shape().as_list()[1]
        kxy = tf.placeholder(tf.float32, shape=[self.num_samples_params, self.num_samples_params])
        dxkxy = tf.placeholder(tf.float32, shape=[self.num_samples_params, params_dim])
        grad_theta = (tf.matmul(kxy, -grad_func) + dxkxy) / self.num_samples_params
        step_size = tf.constant(step_size_value, dtype=tf.float32)

        if self.opt_name == 'ada':
            fudge_factor = tf.constant(fudge_factor_value, dtype=tf.float32)
            alpha = tf.constant(alpha_value, dtype=tf.float32)
            if self.linear and self.sparse:
                params_assign = []

                for ip in range(len(params)):
                    historical_grad = tf.Variable(np.zeros(params_dim), dtype=tf.float32)
                    historical_grad_assign = tf.assign(
                        historical_grad,
                        tf.cond(
                            self.current_iter > 0,
                            lambda: alpha * historical_grad + (1 - alpha) * tf.pow(grad_theta[ip, :], 2.0),
                            lambda: tf.pow(grad_theta[ip, :], 2.0)
                        )
                    )
                    adj_grad = tf.divide(grad_theta[ip, :], fudge_factor + tf.sqrt(historical_grad_assign))

                    params_assign.append(tf.assign(params[ip], params[ip] + step_size * adj_grad))
            else:
                historical_grad = tf.Variable(np.zeros((self.num_samples_params, params_dim)), dtype=tf.float32)

                historical_grad_assign = tf.assign(
                    historical_grad,
                    tf.cond(
                        self.current_iter > 0,
                        lambda: alpha * historical_grad + (1 - alpha) * tf.pow(grad_theta, 2.0),
                        lambda: tf.pow(grad_theta, 2.0)
                    )
                )
                adj_grad = tf.divide(grad_theta, fudge_factor + tf.sqrt(historical_grad_assign))

                params_assign = tf.assign(params, params + step_size * adj_grad)
        else:
            params_assign = tf.assign(params, params + step_size * grad_theta)
        return params_assign, kxy, dxkxy

    def _update_vars(
            self, it, update_operations, kxy_holder, dxkxy_holder, params_value, kernel_params_width, idx_samples):
        kxy, dxkxy = self.svgd_kernel(params_value, kernel_params_width)
        if self.sparse:
            x_feed_tmp = self.x_[idx_samples, :].tocoo()
            feed_data = {
                self.x: (
                    np.array([x_feed_tmp.row, x_feed_tmp.col]).T,
                    x_feed_tmp.data),
                self.y: self.y_[idx_samples],
                kxy_holder: kxy,
                dxkxy_holder: dxkxy,
                self.current_iter: it,
            }
            if not self.linear:
                feed_data[self.eps_omega] = self.eps_omega_const
        else:
            feed_data = {
                self.x: self.x_[idx_samples, :],
                self.y: self.y_[idx_samples],
                kxy_holder: kxy,
                dxkxy_holder: dxkxy,
                self.current_iter: it,
            }
            if not self.linear:
                feed_data[self.eps_omega] = self.eps_omega_const
        new_value = self.tf_session.run(update_operations, feed_dict=feed_data)
        if self.linear and self.sparse:
            new_value = np.vstack(new_value)
        return new_value

    def svgd_kernel(self, params, kernel_params_width):
        sq_dist = pdist(params)
        pairwise_dists = squareform(sq_dist)**2
        if kernel_params_width < 0:  # if h < 0, using median trick
            kernel_params_width = np.median(pairwise_dists)
            kernel_params_width = np.sqrt(0.5 * kernel_params_width / np.log(params.shape[0]+1))
            self.cur_param_kernel_width = kernel_params_width
            if self.info > 1:
                print('kernel_params_width:', kernel_params_width)

        # compute the rbf kernel
        kxy = np.exp(-pairwise_dists / kernel_params_width**2 / 2)

        dxkxy = -np.matmul(kxy, params)
        sumkxy = np.sum(kxy, axis=1)
        for i in range(params.shape[1]):
            dxkxy[:, i] = \
                dxkxy[:, i] + \
                np.multiply(params[:, i], sumkxy)
        dxkxy = dxkxy / (kernel_params_width**2)
        return kxy, dxkxy

    def _fit_loop(self, x, y,
                  do_validation=False,
                  x_valid=None, y_valid=None,
                  callbacks=None, callback_metrics=None):
        pass

    def update_forward(self, x, y=None):
        num_tests = x.shape[0]
        y_pred = np.ones(num_tests, dtype=int)
        loss = 0

        # self._assign_params_value()

        for it in range(0, num_tests, self.batch_size):
            if self.sparse:
                x_batch = scipy.sparse.lil_matrix((self.batch_size, self.input_dim), dtype=float)
            else:
                x_batch = np.zeros((self.batch_size, self.input_dim))
            y_batch = np.zeros(self.batch_size)
            idx_end_dataset = np.minimum(it + self.batch_size, num_tests)
            idx_end_batch = idx_end_dataset - it
            x_batch[0:idx_end_batch] = x[it:idx_end_dataset, :]
            if y is not None:
                y_batch[0:idx_end_batch] = y[it:idx_end_dataset]

            if self.sparse:
                x_feed_tmp = x_batch.tocoo()
                feed_data = {
                    self.x: (
                        np.array([x_feed_tmp.row, x_feed_tmp.col]).T,
                        x_feed_tmp.data),
                    self.y: y_batch,
                }
                if not self.linear:
                    feed_data[self.eps_omega] = self.eps_omega_const
            else:
                feed_data = {
                    self.x: x_batch,
                    self.y: y_batch,
                }
                if not self.linear:
                    feed_data[self.eps_omega] = self.eps_omega_const
            loss_tmp, wx_tmp = self.tf_session.run([self.loss_all, self.wx_all], feed_dict=feed_data)
            loss_tmp = np.sum(loss_tmp)
            if self.predict_mode == 0:
                wx_tmp = np.sum(wx_tmp, axis=0)
                y_tmp = np.argmax(wx_tmp, axis=1)
            else:
                y_tmp = scipy.stats.mode(np.argmax(wx_tmp, axis=2), axis=0)[0][0]

            loss += loss_tmp
            y_pred[it:idx_end_dataset] = y_tmp[0:idx_end_batch]

        loss /= num_tests

        return loss, y_pred

    def predict(self, x):
        _, y = self.update_forward(x)
        return self._decode_labels(y)

    def display_prediction(self, **kwargs):
        visualize_classification_prediction(self, self.x_, self.y_, **kwargs)

    def display(self, param, **kwargs):
        if param == 'predict':
            self.display_prediction(**kwargs)
        else:
            raise NotImplementedError

    def _on_train_end(self):
        if self.info > 0:
            print('\nFinish training')
        super(BaseBKM, self)._on_train_end()

    def get_params(self, deep=True):
        out = super(BaseBKM, self).get_params(deep=deep)
        param_names = BaseBKM._get_param_names()
        out.update(self._get_params(param_names=param_names, deep=deep))
        return out
