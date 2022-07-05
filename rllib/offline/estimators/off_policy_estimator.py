import logging
from ray.rllib.policy.sample_batch import MultiAgentBatch, DEFAULT_POLICY_ID
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.typing import TensorType, SampleBatchType
from typing import Dict, Any
from ray.rllib.utils.numpy import convert_to_numpy

logger = logging.getLogger(__name__)


def action_log_likelihood(policy: Policy, batch: SampleBatch):
    """Returns log likelihood for actions in given batch for policy.

    Computes likelihoods by passing the observations through the current
    policy's `compute_log_likelihoods()` method

    Args:
        batch: The SampleBatch or MultiAgentBatch to calculate action
            log likelihoods from. This batch/batches must contain OBS
            and ACTIONS keys.

    Returns:
        The probabilities of the actions in the batch, given the
        observations and the policy.
    """
    num_state_inputs = 0
    for k in batch.keys():
        if k.startswith("state_in_"):
            num_state_inputs += 1
    state_keys = ["state_in_{}".format(i) for i in range(num_state_inputs)]
    log_likelihoods: TensorType = policy.compute_log_likelihoods(
        actions=batch[SampleBatch.ACTIONS],
        obs_batch=batch[SampleBatch.OBS],
        state_batches=[batch[k] for k in state_keys],
        prev_action_batch=batch.get(SampleBatch.PREV_ACTIONS),
        prev_reward_batch=batch.get(SampleBatch.PREV_REWARDS),
        actions_normalized=policy.config["actions_in_input_normalized"],
    )
    log_likelihoods = convert_to_numpy(log_likelihoods)
    return log_likelihoods


@DeveloperAPI
class OffPolicyEstimator:
    """Interface for an off policy reward estimator."""

    @DeveloperAPI
    def __init__(self, name: str, policy: Policy, gamma: float):
        """Initializes an OffPolicyEstimator instance.

        Args:
            name: string to save OPE results under
            policy: Policy to evaluate.
            gamma: Discount factor of the environment.
        """
        self.name = name
        self.policy = policy
        self.gamma = gamma

    @DeveloperAPI
    def estimate(self, batch: SampleBatchType) -> Dict[str, Any]:
        """Returns off policy estimates for the given batch of episodes.

        Args:
            batch: The batch to calculate the off policy estimates (OPE) on.

        Returns:
            The off-policy estimates (OPE) calculated on the given batch. Note
            that this dictionary can contain arbitrary strings mapping to the
            list of metrics e.g. {"v_old": [1, 2, 3], "v_new": [4,5,6]}
        """
        raise NotImplementedError

    @DeveloperAPI
    def check_can_estimate_for(self, batch: SampleBatchType) -> None:
        """Checks if we support off policy estimation (OPE) on given batch.

        Args:
            batch: The batch to check.

        Raises:
            ValueError: In case `action_prob` key is not in batch OR batch
            is a MultiAgentBatch.
        """

        if isinstance(batch, MultiAgentBatch):
            policy_keys = batch.policy_batches.keys()
            if len(policy_keys) == 1 and DEFAULT_POLICY_ID in policy_keys:
                batch = batch.policy_batches[DEFAULT_POLICY_ID]
            else:
                raise ValueError(
                    "Off-Policy Estimation is not implemented for "
                    "multi-agent batches. You can set "
                    "`off_policy_estimation_methods: {}` to resolve this."
                )

        if "action_prob" not in batch:
            raise ValueError(
                "Off-policy estimation is not possible unless the inputs "
                "include action probabilities (i.e., the policy is stochastic "
                "and emits the 'action_prob' key). For DQN this means using "
                "`exploration_config: {type: 'SoftQ'}`. You can also set "
                "`off_policy_estimation_methods: {}` to disable estimation."
            )

    @DeveloperAPI
    def train(self, batch: SampleBatchType) -> Dict[str, Any]:
        """Train a model for Off-Policy Estimation.

        Args:
            batch: SampleBatch to train on

        Returns:
            Any optional metrics to return from the estimator
        """
        return {}

    @Deprecated(
        old="OffPolicyEstimator.action_log_likelihood",
        new="ray.rllib.offline.estimators.off_policy_estimator.action_log_likelihood",
        error=False,
    )
    def action_log_likelihood(self, batch: SampleBatchType) -> TensorType:
        return action_log_likelihood(self.policy, batch)
