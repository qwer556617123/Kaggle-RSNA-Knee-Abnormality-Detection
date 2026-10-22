"""Keep synthetic neural-model tests small and deterministic on large CPU hosts."""
import torch

torch.set_num_threads(2)
