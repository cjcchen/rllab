from rllab.misc.console import mkdir_p

import tensorflow as tf
import os


class Summary:
    def __init__(self):
        self._summary_scale = tf.Summary()
        self._histogram_ds = {}
        self._histogram_summary_op = []

        self._session = tf.Session()

        self._default_step = 0
        self._step_key = None
        self._writer = None

    def record_histogram(self, key, val):
        if str(key) not in self._histogram_ds:
            self._histogram_ds[str(key)] = tf.Variable(val)
            self._histogram_summary_op.append(
                tf.summary.histogram(str(key), self._histogram_ds[str(key)]))

        x = tf.assign(self._histogram_ds[str(key)], val)
        self._session.run(x)

    def record_scale(self, key, val):
        self._summary_scale.value.add(tag=str(key), simple_value=float(val))

    def dump_tensorboard(self, step=None):
        run_step = self._default_step
        if step:
            run_step = step
        else:
            self._default_step += 1

        self._dump_histogram(run_step)
        self._dump_scale(run_step)

    def set_dir(self, dir_name):
        if not dir_name:
            if self._writer:
                self._writer.close()
                self._writer = None
        else:
            mkdir_p(os.path.dirname(dir_name))
            self._writer = tf.summary.FileWriter(dir_name)
            self._default_step = 0
            assert self._writer is not None
            print("tensorboard data will be logged into:", dir_name)

    def _dump_histogram(self, step):
        if len(self._histogram_summary_op):
            summary_str = self._session.run(
                tf.summary.merge(self._histogram_summary_op))
            self._writer.add_summary(summary_str, global_step=step)
            self._writer.flush()

    def _dump_scale(self, step):
        self._writer.add_summary(self._summary_scale, step)
        self._writer.flush()
        del self._summary_scale.value[:]
