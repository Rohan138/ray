from typing import Dict
from ray.rllib.offline.estimators.off_policy_estimator import (
    OffPolicyEstimator,
    OffPolicyEstimate,
)
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.typing import SampleBatchType
from ray.rllib.offline.estimators.qreg_torch_model import QRegTorchModel
from gym.spaces import Discrete
import numpy as np


def k_fold_cv(batch: SampleBatchType, k: int):
    """Utility function that returns a k-fold cross validation generator
    over episodes from the given batch.

    Args:
        batch: A SampleBatch of episodes to split
        k: Number of cross-validation splits

    Returns:
        A tuple of SampleBatches (train_episodes, test_episodes)
    """
    episodes = batch.split_by_episode()
    n_episodes = len(episodes)
    if n_episodes < k:
        # TODO(rohan): print warning: "len(batch) < k, running OPE without training"
        yield [], episodes
        return
    n_fold = n_episodes // k
    for i in range(k):
        train_episodes = episodes[: i * n_fold] + episodes[(i + 1) * n_fold :]
        if i != k - 1:
            test_episodes = episodes[i * n_fold : (i + 1) * n_fold]
        else:
            # Append remaining episodes onto the last test_episodes
            test_episodes = episodes[i * n_fold :]
        yield train_episodes, test_episodes
    return


class DirectMethod(OffPolicyEstimator):
    """The Direct Method (Q-Reg) estimator.

    config: {
        model: ModelConfigDict,
        k: k-fold cross validation for training model and evaluating OPE
    }

    Q-Reg estimator described in https://arxiv.org/pdf/1511.03722.pdf,
    https://arxiv.org/pdf/1911.06854.pdf"""

    @override(OffPolicyEstimator)
    def __init__(self, policy: Policy, gamma: float, config: Dict):
        super().__init__(policy, gamma, config)
        assert isinstance(
            policy.action_space, Discrete
        ), "DM Estimator only supports discrete action spaces!"
        assert (
            policy.config["batch_mode"] == "complete_episodes"
        ), "DM Estimator only supports batch_mode=`complete_episodes`"
        assert (
            policy.framework == "torch"
        ), "DM estimator only supports `framework`=`torch`"
        # TODO (rohan): Add support for QRegTF, FQETorch, FQETF
        model_cls = QRegTorchModel
        self.model = model_cls(
            policy=policy,
            gamma=gamma,
            config=config,
        )
        self.k = config.get("k", 5)

    @override(OffPolicyEstimator)
    def estimate(self, batch: SampleBatchType) -> OffPolicyEstimate:
        self.check_can_estimate_for(batch)
        estimates = []
        # Split data into train and test using k-fold cross validation
        for train_episodes, test_episodes in k_fold_cv(batch, self.k):
            # Reinitialize model
            self.model.reset()

            # Train Q-function
            if train_episodes:
                train_batch = train_episodes[0].concat_samples(train_episodes)
                losses = self.train(train_batch)  # noqa: F841

            # Calculate direct method OPE estimates
            for episode in test_episodes:
                rewards = episode["rewards"]
                v_old = 0.0
                v_dm = 0.0
                for t in range(episode.count):
                    v_old += rewards[t] * self.gamma ** t

                init_step = episode[0:1]
                init_obs = np.array([init_step[SampleBatch.OBS]])
                all_actions = np.array(
                    [a for a in range(self.policy.action_space.n)], dtype=float
                )
                init_step[SampleBatch.ACTIONS] = all_actions
                action_probs = np.exp(self.action_log_likelihood(init_step))
                v_value = self.model.estimate_v(init_obs, action_probs)
                v_dm = convert_to_numpy(v_value).item()

                estimates.append(
                    OffPolicyEstimate(
                        "direct_method",
                        {
                            "v_old": v_old,
                            "v_dm": v_dm,
                            "v_gain": v_dm / max(1e-8, v_old),
                        },
                    )
                )
        return estimates

    @override(OffPolicyEstimator)
    def train(self, batch: SampleBatchType):
        return self.model.train_q(batch)
