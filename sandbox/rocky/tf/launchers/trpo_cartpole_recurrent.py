from sandbox.rocky.tf.algos import TRPO
from rllab.baselines import LinearFeatureBaseline
from rllab.envs.box2d import CartpoleEnv
from rllab.envs import normalize
from sandbox.rocky.tf.policies import GaussianGRUPolicy
from sandbox.rocky.tf.policies import GaussianLSTMPolicy
from sandbox.rocky.tf.envs import TfEnv
import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.optimizers import ConjugateGradientOptimizer, FiniteDifferenceHvp
from rllab.misc import stub, run_experiment_lite

env = TfEnv(normalize(CartpoleEnv()))

policy = GaussianLSTMPolicy(
    name="policy",
    env_spec=env.spec,
    lstm_layer_cls=L.TfBasicLSTMLayer,
    # gru_layer_cls=L.GRULayer,
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=4000,
    max_path_length=100,
    n_itr=10,
    discount=0.99,
    step_size=0.01,
    optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
)
algo.train()
