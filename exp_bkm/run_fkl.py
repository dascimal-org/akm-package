from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from male.models.kernel.fkl import FKL
from male.test_template import main_func, resolve_conflict_params

import numpy as np

choice_default = 2


def create_obj_func(params):
    if choice_default == 0:
        default_params = {
            'D': 400,
            'lbd': 0.001,
            'gamma': 1.0,
            'loss': 'hinge',
            'mode': 'batch',
            'num_epochs': 10,
            'learning_rate': 0.001,
            'learning_rate_gamma': 0.001,
            'catch_exception': True,
            'random_state': 1010,
        }
    else:
        default_params = {
            'D': 400,
            'lbd':  3.0517578125e-05,
            'gamma': 0.5,
            'batch_size': 200,
            'loss': 'hinge',
            'mode': 'batch',
            'num_epochs': 10,
            'learning_rate': 0.001,
            'learning_rate_gamma': 0.001,
            'catch_exception': True,
        }

    default_params = resolve_conflict_params(params, default_params)
    print('default_params:', default_params)

    learner = FKL(
        **params,
        **default_params,
    )
    return learner


def main_test(run_exp=False):
    params_gridsearch = {
        'gamma': 2.0 ** np.arange(-5, 15, 2),
        'lbd': 2.0 ** np.arange(-15, 5, 2),
        'learning_rate_gamma': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0],
        'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0],
    }
    attribute_names = (
        'gamma', 'D', 'learning_rate', 'learning_rate_gamma', 'lbd', 'num_epochs', 'model_name', 'loss')

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
        keep_vars=['e', 'b'],
    )


if __name__ == '__main__':
    # pytest.main([__file__])
    main_test(run_exp=True)
