from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from male.models.kernel.bkm import BKM
from male.test_template import main_func, resolve_conflict_params

choice_default = 2


def create_obj_func(params):
    if choice_default == 0:
        default_params = {
            'w_regular': 0.1,
            'rf_dim': 400,
            'step_size': 0.5,
            'x_kernel_width': 0.0001,
            'params_kernel_width': 15,
            'num_iters': 500,
            'num_samples_params': 100,
            'batch_size': 100,
            'freq_calc_metrics': 1,
        }
    else:
        default_params = {
            'w_regular': 1e-3,
            'rf_dim': 400,
            'step_size': 1e-3,
            'params_kernel_width': -1,
            'num_samples_params': 100,
            'batch_size': 200,
            'freq_calc_metrics': 1,
            'info': 1
        }

    default_params = resolve_conflict_params(params, default_params)
    print('Default params:', default_params)

    learner = BKM(
        **params,
        **default_params,
    )
    return learner


def main_test(run_exp=False):
    params_gridsearch = {
        'w_regular': [1e-3],
        'step_size': [1e-3],
    }
    attribute_names = (
        'w_regular', 'step_size', 'x_kernel_width', 'batch_size',
        'num_iters', 'rf_dim', 'compact_radius', 'model_name', 'num_samples_params', 'params_kernel_width',
        'scale_w', 'opt_name', 'cur_param_kernel_width')

    main_func(
        create_obj_func,
        choice_default=choice_default,
        dataset_default='svmguide1',
        params_gridsearch=params_gridsearch,
        attribute_names=attribute_names,
        num_workers=4,
        file_config=None,
        run_exp=run_exp,
        freq_predict_display=10,
    )


if __name__ == '__main__':
    # pytest.main([__file__])
    main_test(run_exp=True)
