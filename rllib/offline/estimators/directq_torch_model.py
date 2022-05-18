import gym
from gym.spaces import Box, Discrete
import numpy as np
from typing import Dict, List, Optional

from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType, TensorStructType
torch, nn = try_import_torch()


class DirectQTorchModel(TorchModelV2, nn.Module):
    """Extension of the standard TorchModelV2 for Q-function estimation.
    """

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: Optional[int],
        model_config: ModelConfigDict,
        name: str,
    ):
        """Initializes a DirectQTorchModel instance.
        """
        nn.Module.__init__(self)
        super(DirectQTorchModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        if isinstance(action_space, Discrete):
            self.action_dim = action_space.n
            self.discrete = True
        elif isinstance(action_space, Box):
            self.action_dim = np.product(action_space.shape)
            self.discrete = False
            num_outputs = 1
        else:
            raise ValueError("Estimator only supports Box and Discrete action spaces!")

        self.concat_obs_and_actions = False
        if self.discrete:
            input_space = obs_space
        else:
            orig_space = getattr(obs_space, "original_space", obs_space)
            if isinstance(orig_space, Box) and len(orig_space.shape) == 1:
                input_space = Box(
                    float("-inf"),
                    float("inf"),
                    shape=(orig_space.shape[0] + action_space.shape[0],),
                )
                self.concat_obs_and_actions = True
            else:
                input_space = gym.spaces.Tuple([orig_space, action_space])
        
        self.q_model = ModelCatalog.get_model_v2(
            input_space,
            action_space,
            num_outputs,
            model_config,
            framework="torch",
            name=name,
        )

    def get_q_values(self, obs, actions):
        # obs may come as original Tuple observations, concat them
        # here if this is the case.
        if isinstance(self.q_model.obs_space, Box):
            if isinstance(obs, (list, tuple)):
                obs = torch.cat(obs, dim=-1)
            elif isinstance(obs, dict):
                obs = torch.cat(list(obs.values()), dim=-1)

        # Continuous case -> concat actions to obs.
        if actions is not None:
            if self.concat_obs_and_actions:
                input_dict = {"obs": torch.cat([obs, actions], dim=-1)}
            else:
                input_dict = {"obs": (obs, actions)}
        # Discrete case -> return q-vals for all actions.
        else:
            input_dict = {"obs": obs}
        # Switch on training mode (when getting Q-values, we are usually in
        # training).
        input_dict["is_training"] = True

        return self.q_model(input_dict, [], None)

    def variables(self):
        """Return the list of variables for QModel."""

        return self.q_model.variables()
