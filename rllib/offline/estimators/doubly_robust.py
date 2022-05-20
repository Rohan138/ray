from ray.rllib.offline.estimators.off_policy_estimator import OffPolicyEstimate
from ray.rllib.offline.estimators.direct_method import DirectMethod, k_fold_cv
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import SampleBatchType
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.numpy import convert_to_numpy
import numpy as np


class DoublyRobust(DirectMethod):
    """The Doubly Robust (DR) estimator with Q-Reg Q-function.

    DR estimator described in https://arxiv.org/pdf/1511.03722.pdf,
    Q-Reg DR in https://arxiv.org/pdf/1911.06854.pdf"""

    @override(DirectMethod)
    def estimate(self, batch: SampleBatchType) -> OffPolicyEstimate:
        self.check_can_estimate_for(batch)
        estimates = []
        # Split data into train and test using k-fold cross validation
        for train_episodes, test_episodes in k_fold_cv(batch, self.k):
            # Reinitialize model
            self.model.reset()

            # Train Q-function
            train_batch = train_episodes[0].concat_samples(train_episodes)
            # TODO (rohan): log the training loss somewhere
            losses = self.train(train_batch)  # noqa: F841

            # Calculate doubly robust OPE estimates
            for episode in test_episodes:
                rewards, old_prob = episode["rewards"], episode["action_prob"]
                new_prob = np.exp(self.compute_log_likelihoods(episode))

                V_prev, V_DR = 0.0, 0.0
                v_values = self.model.estimate_v(episode[SampleBatch.OBS])
                v_values = convert_to_numpy(v_values)
                q_values = self.model.estimate_q(
                    episode[SampleBatch.OBS], episode[SampleBatch.ACTIONS]
                )
                q_values = convert_to_numpy(q_values)
                for t in range(episode.count, -1, -1):
                    V_prev = rewards[t] + self.gamma * V_prev
                    V_DR = v_values[t] + (new_prob[t] / old_prob[t]) * (
                        rewards[t] + self.gamma * V_DR - q_values[t]
                    )

                estimates.append(
                    OffPolicyEstimate(
                        "doubly_robust",
                        {
                            "V_prev": V_prev,
                            "V_DR": V_DR,
                            "V_gain_est": V_DR / max(1e-8, V_prev),
                        },
                    )
                )
        return estimates
