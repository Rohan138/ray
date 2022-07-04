import cupy as cp
import numpy as np
from ray.rllib.utils.framework import try_import_torch, try_import_tf, try_import_jax
from ray.rllib.utils.typing import TensorStructType

torch, _ = try_import_torch()
tf1, tf2, _ = try_import_tf()
jax, _ = try_import_jax()

def convert_to_dlpack(x: TensorStructType):
    pass

def convert_from_dlpack(x: TensorStructType, framework = None):
    pass