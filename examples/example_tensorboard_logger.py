from rllab.misc import logger

import numpy as np

logger.set_tensorboard_dir("./test")

for i in range(100):
    val = np.random.normal(0, i, size=(3, 3, 3))
    logger.record_tabular("app", i)
    logger.record_histogram("gass", val)
    logger.dump_tabular()
