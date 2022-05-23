import unittest
import ray
from ray.rllib.algorithms.dqn import DQNTrainer
from ray.rllib.offline.estimators import (
    ImportanceSampling,
    WeightedImportanceSampling,
    DirectMethod,
    DoublyRobust,
)
from ray.rllib.offline.json_reader import JsonReader
from pathlib import Path
import os


class TestOPE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ray.init(num_cpus=4)

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()

    def test_build_ope_methods(self):
        rllib_dir = Path(__file__).parent.parent.parent
        print("rllib dir={}".format(rllib_dir))
        data_file = os.path.join(rllib_dir, "tests/data/cartpole/small.json")
        print("data_file={} exists={}".format(data_file, os.path.isfile(data_file)))

        trainer = DQNTrainer(
            env="CartPole-v0",
            config={
                "input": data_file,
                "input_evaluation": [],
                "framework": "torch",
                "exploration_config": {
                    "type": "SoftQ",
                    "temperature": 1.0,
                },
            },
        )
        trainer.train()

        estimators = [
            ImportanceSampling,
            WeightedImportanceSampling,
            DirectMethod,
            DoublyRobust,
        ]
        for estimator_cls in estimators:
            estimator = estimator_cls(
                trainer.get_policy(),
                gamma=0.99,
                config={"k": 5, "n_iters": 600, "lr": 1e-3, "delta": 1e-5},
            )
            reader = JsonReader(data_file)
            batch = reader.next()
            for _ in range(10):
                batch = batch.concat(reader.next())
            estimate = estimator.estimate(batch)
            assert len(estimate) == len(batch.split_by_episode())
            print(estimate)


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(["-v", __file__]))
