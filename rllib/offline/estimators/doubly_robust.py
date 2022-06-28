from ray.rllib.offline.estimators.off_policy_estimator import OffPolicyEstimator
from ray.rllib.offline.estimators.utils import (
    lookup_state_value_fn,
    lookup_action_value_fn,
)
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.typing import SampleBatchType, TensorType
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy import Policy
from ray.rllib.utils.numpy import convert_to_numpy
import numpy as np
from typing import Dict, Any, Callable


@DeveloperAPI
class DoublyRobust(OffPolicyEstimator):
    """The Doubly Robust (DR) estimator.

    DR estimator described in https://arxiv.org/pdf/1511.03722.pdf"""

    @override(OffPolicyEstimator)
    def __init__(
        self,
        name: str,
        policy: Policy,
        gamma: float,
        state_value_fn: Callable[[Policy, SampleBatch], TensorType] = None,
        action_value_fn: Callable[[Policy, SampleBatch], TensorType] = None,
    ):
        """
        Initializes a Direct Method OPE Estimator.

        Args:
            name: string to save OPE results under
            policy: Policy to evaluate.
            gamma: Discount factor of the environment.
            state_value_fn: Function that takes in self.policy and a
            SampleBatch with states s and return the state values V(s).
            This is meant to be generic; modify this for your Algorithm as neccessary.
            If not specified, try to look up the function using lookup_state_value_fn.
            action_value_fn: Function that takes in self.policy and a
            SampleBatch with states s and actions a and return the action values Q(s,a).
            This is meant to be generic; modify this for your Algorithm as neccessary.
            If not specified, try to look up the function using lookup_action_value_fn.
        """

        super().__init__(name, policy, gamma)
        self.state_value_fn = state_value_fn or lookup_state_value_fn(policy)
        self.action_value_fn = action_value_fn or lookup_action_value_fn(policy)

    @override(OffPolicyEstimator)
    def estimate(self, batch: SampleBatchType) -> Dict[str, Any]:
        self.check_can_estimate_for(batch)
        estimates = {"v_old": [], "v_new": [], "v_gain": []}
        # Calculate doubly robust OPE estimates
        for episode in batch.split_by_episode():
            rewards, old_prob = episode["rewards"], episode["action_prob"]
            new_prob = np.exp(self.action_log_likelihood(episode))

            v_old = 0.0
            v_new = 0.0
            q_values = self
            q_values = convert_to_numpy(q_values)

            all_actions = np.zeros([episode.count, self.policy.action_space.n])
            all_actions[:] = np.arange(self.policy.action_space.n)
            # Two transposes required for torch.distributions to work
            tmp_episode = episode.copy()
            tmp_episode[SampleBatch.ACTIONS] = all_actions.T
            action_probs = np.exp(self.action_log_likelihood(tmp_episode)).T
            v_values = self.model.estimate_v(episode[SampleBatch.OBS], action_probs)
            v_values = convert_to_numpy(v_values)

            for t in reversed(range(episode.count)):
                v_old = rewards[t] + self.gamma * v_old
                v_new = v_values[t] + (new_prob[t] / old_prob[t]) * (
                    rewards[t] + self.gamma * v_new - q_values[t]
                )
            v_new = v_new.item()

            estimates["v_old"].append(v_old)
            estimates["v_new"].append(v_new)
            estimates["v_gain"].append(v_new / max(v_old, 1e-8))
        estimates["v_old_std"] = np.std(estimates["v_old"])
        estimates["v_old"] = np.mean(estimates["v_old"])
        estimates["v_new_std"] = np.std(estimates["v_new"])
        estimates["v_new"] = np.mean(estimates["v_new"])
        estimates["v_gain_std"] = np.std(estimates["v_gain"])
        estimates["v_gain"] = np.mean(estimates["v_gain"])
        return estimates
