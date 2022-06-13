import os
import unittest
from pathlib import Path

import numpy as np

import ray
from ray.rllib.algorithms import cql
from ray.rllib.offline.estimators.importance_sampling import ImportanceSampling
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import (
    check_compute_single_action,
    check_train_results,
    framework_iterator,
)

tf1, tf, tfv = try_import_tf()
torch, _ = try_import_torch()


class TestCQL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ray.init()

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()

    def test_cql_compilation(self):
        """Test whether CQL can be built with all frameworks."""

        # Learns from a historic-data file.
        # To generate this data, first run:
        # $ ./train.py --run=SAC --env=Pendulum-v1 \
        #   --stop='{"timesteps_total": 50000}' \
        #   --config='{"output": "/tmp/out"}'
        rllib_dir = Path(__file__).parent.parent.parent.parent
        print("rllib dir={}".format(rllib_dir))
        data_file = os.path.join(rllib_dir, "tests/data/pendulum/small.json")
        print("data_file={} exists={}".format(data_file, os.path.isfile(data_file)))

        config = (
            cql.CQLConfig()
            .environment(
                env="Pendulum-v1",
            )
            .offline_data(
                input_=data_file,
                # In the files, we use here for testing, actions have already
                # been normalized.
                # This is usually the case when the file was generated by another
                # RLlib algorithm (e.g. PPO or SAC).
                actions_in_input_normalized=False,
            )
            .training(
                clip_actions=False,
                train_batch_size=2000,
                twin_q=True,
                replay_buffer_config={"learning_starts": 0},
                bc_iters=2,
            )
            .evaluation(
                always_attach_evaluation_results=True,
                evaluation_interval=2,
                evaluation_duration=10,
                evaluation_config={"input": "sampler"},
                evaluation_parallel_to_training=False,
                evaluation_num_workers=2,
                # Switch on off-policy evaluation.
                off_policy_estimation_methods={"is": {"type": ImportanceSampling}},
            )
            .rollouts(rollout_fragment_length=1)
        )
        num_iterations = 4

        # Test for tf/torch frameworks.
        for fw in framework_iterator(config, with_eager_tracing=True):
            trainer = config.build()
            for i in range(num_iterations):
                results = trainer.train()
                check_train_results(results)
                print(results)
                eval_results = results["evaluation"]
                print(
                    f"iter={trainer.iteration} "
                    f"R={eval_results['episode_reward_mean']}"
                )

            check_compute_single_action(trainer)

            # Get policy and model.
            pol = trainer.get_policy()
            cql_model = pol.model
            if fw == "tf":
                pol.get_session().__enter__()

            # Example on how to do evaluation on the trained Trainer
            # using the data from CQL's global replay buffer.
            # Get a sample (MultiAgentBatch).
            multi_agent_batch = trainer.local_replay_buffer.sample(
                num_items=config.train_batch_size
            )
            # All experiences have been buffered for `default_policy`
            batch = multi_agent_batch.policy_batches["default_policy"]

            if fw == "torch":
                obs = torch.from_numpy(batch["obs"])
            else:
                obs = batch["obs"]
                batch["actions"] = batch["actions"].astype(np.float32)

            # Pass the observations through our model to get the
            # features, which then to pass through the Q-head.
            model_out, _ = cql_model({"obs": obs})
            # The estimated Q-values from the (historic) actions in the batch.
            if fw == "torch":
                q_values_old = cql_model.get_q_values(
                    model_out, torch.from_numpy(batch["actions"])
                )
            else:
                q_values_old = cql_model.get_q_values(
                    tf.convert_to_tensor(model_out), batch["actions"]
                )

            # The estimated Q-values for the new actions computed
            # by our trainer policy.
            actions_new = pol.compute_actions_from_input_dict({"obs": obs})[0]
            if fw == "torch":
                q_values_new = cql_model.get_q_values(
                    model_out, torch.from_numpy(actions_new)
                )
            else:
                q_values_new = cql_model.get_q_values(model_out, actions_new)

            if fw == "tf":
                q_values_old, q_values_new = pol.get_session().run(
                    [q_values_old, q_values_new]
                )

            print(f"Q-val batch={q_values_old}")
            print(f"Q-val policy={q_values_new}")

            if fw == "tf":
                pol.get_session().__exit__(None, None, None)

            trainer.stop()


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main(["-v", __file__]))
