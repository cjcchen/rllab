from rllab.misc import logger

import numpy as np

logger.set_tensorboard_dir("./histogram_example2")
N = 400
for i in range(N):
    k_val = i/float(N)
    logger.record_histogram_by_type('gamma', key = 'gamma', alpha=k_val)
    logger.record_histogram_by_type('normal', key = 'normal', mean=5*k_val, stddev=1.0)
    logger.record_histogram_by_type('poisson', key = 'poisson', lam=k_val)
    logger.record_histogram_by_type('uniform', key = 'uniform', maxval=k_val*10)
    logger.record_tabular("app", k_val)
    logger.record_histogram("gass", k_val)
    logger.dump_tensorboard(step=i)

