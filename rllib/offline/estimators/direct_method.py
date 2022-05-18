from ray.rllib.offline.estimators.off_policy_estimator import OffPolicyEstimator, OffPolicyEstimate
from ray.rllib.offline
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import SampleBatchType, ModelConfigDict
from ray.rllib.policy import Policy
from ray.rllib.models import ModelCatalog
from ray.rllib.offline.estimators.directq_tf_model import DirectQTFModel
from ray.rllib.offline.estimators.directq_torch_model import DirectQTorchModel

class DirectMethod(OffPolicyEstimator):
    """The Direct Method estimator, implemented using ray.rllib.models.ModelV2.

    DM estimator described in https://arxiv.org/pdf/1511.03722.pdf"""

    @override(OffPolicyEstimator)
    def __init__(self, policy: Policy, gamma: float, q_model_config: ModelConfigDict):
        super().__init__(policy, gamma)

        # Infer input_space and num_outputs for Q(s,a).
        # If the action space is Discrete, we input the state s to the model
        # and output a vector Q(s,a) for all actions a in the action space.
        # If the action space is Box, we concatenate and input s|a to the model
        # and output a single value Q(s,a).
        self.model = ModelCatalog.get_model_v2(
            obs_space=policy.observation_space,
            action_space=policy.action_space,
            num_outputs=1, # Arbitrary; the num_outputs is directly inferred similar to SAC
            model_config=q_model_config,
            framework=policy.framework,
            model_interface=DirectQTorchModel if policy.framework=="torch" else DirectQTFModel,
            name="DirectQModel",
        )

    @override(OffPolicyEstimator)
    def estimate(self, batch: SampleBatchType) -> OffPolicyEstimate:
        # Note: DM can technically work without action_prob,
        # but none of the other methods implemented so far can
        self.check_can_estimate_for(batch)

        # calculate Direct Method estimate
        rewards = batch["rewards"]
        V_prev = 0.0
        for t in range(batch.count):
            V_prev += rewards[t] * self.gamma ** t
        actions = self.policy.compute_actions(batch["obs"])
        values = self.model.get_q_values(batch["obs"], actions)
        V_DM = V_DM[0]
        if self.policy.framework == "torch":
            V_DM = V_DM.item()

        estimation = OffPolicyEstimate(
            "direct_method",
            {
                "V_prev": V_prev,
                "V_DM": V_DM,
                "V_gain_est": V_DM / max(1e-8, V_prev),
            },
        )
        return estimation

    @override(OffPolicyEstimator)
    def train_q(self, batch: SampleBatchType):
        self.model.train_q(batch, self.policy, self.gamma)