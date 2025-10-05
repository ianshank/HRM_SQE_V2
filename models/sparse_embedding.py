from typing import Union, Any, Iterable

import torch
from torch import nn
import torch.distributed as dist
from torch.optim.optimizer import Optimizer

from models.common import trunc_normal_init_

# Type alias for compatibility with PyTorch < 2.2
ParamsT = Iterable[torch.Tensor]


class CastedSparseEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, batch_size: int, init_std: float, cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to
        self.embedding_dim = embedding_dim

        # Real Weights
        # Truncated LeCun normal init
        weight_dtype = self.cast_to if self.cast_to in (torch.float16, torch.bfloat16) else torch.float32
        self.register_buffer(
            "weights",
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim), dtype=weight_dtype), std=init_std),
            persistent=True
        )

        # Local weights and IDs - will be resized dynamically
        # Local embeddings, with gradient, not persistent
        self.local_weights = None
        # Local embedding IDs, not persistent
        self.local_ids = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Validate input indices to prevent out-of-bounds errors
        if inputs.max() >= self.weights.shape[0] or inputs.min() < 0:
            raise IndexError(
                f"Input indices out of bounds: min={inputs.min()}, max={inputs.max()}, "
                f"valid range=[0, {self.weights.shape[0]-1}]"
            )

        if not self.training:
            # Test mode, no gradient
            return self.weights[inputs].to(self.cast_to)

        # Training mode, fill puzzle embedding from weights
        batch_size = inputs.shape[0]

        # Create or resize local buffers if needed
        if self.local_weights is None or self.local_weights.shape[0] != batch_size:
            # Create new tensors with the correct size
            self.local_weights = torch.zeros(
                batch_size, self.embedding_dim,
                dtype=torch.float32,
                requires_grad=True,
                device=self.weights.device
            )
            self.local_ids = torch.zeros(
                batch_size,
                dtype=torch.int32,
                device=self.weights.device
            )

        with torch.no_grad():
            self.local_weights.copy_(self.weights[inputs].to(self.local_weights.dtype))
            self.local_ids.copy_(inputs)

        return self.local_weights.to(self.cast_to)


class CastedSparseEmbeddingSignSGD_Distributed(Optimizer):
    def __init__(
        self,
        params: ParamsT,

        world_size: int,
        lr: Union[float, torch.Tensor] = 1e-3,
        weight_decay: float = 1e-2,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            world_size=world_size
        )
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None):  # type: ignore
        for group in self.param_groups:
            # Find the sparse embedding module - should be stored in the group
            sparse_emb_module = group.get("sparse_emb_module")
            if sparse_emb_module is None:
                # Old behavior for backwards compatibility - find weights buffer
                local_weights_grad = None
                local_ids = None
                weights = None

                assert len(group["params"]) == 3
                for p in group["params"]:
                    if p.requires_grad:
                        local_weights_grad = p.grad
                    elif p.ndim == 1:
                        local_ids = p
                    elif p.ndim == 2:
                        weights = p
                    else:
                        assert False

                assert local_weights_grad is not None
                assert local_ids is not None
                assert weights is not None
            else:
                # New behavior - access local_weights and local_ids from module
                if sparse_emb_module.local_weights is None or sparse_emb_module.local_ids is None:
                    # Skip if no gradients have been computed yet
                    continue

                local_weights_grad = sparse_emb_module.local_weights.grad
                if local_weights_grad is None:
                    # Skip if no gradients
                    continue

                local_ids = sparse_emb_module.local_ids
                weights = sparse_emb_module.weights
        
            # Apply SignSGD
            # Adam ≈ SignSGD if gradient is very sparse
            _sparse_emb_signsgd_dist(
                local_weights_grad,
                local_ids,
                weights,
                
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                world_size=group["world_size"]
            )


def _sparse_emb_signsgd_dist(
    local_weights_grad: torch.Tensor,
    local_ids: torch.Tensor,
    weights: torch.Tensor,
    
    lr: float,
    weight_decay: float,
    world_size: int
) -> None:
    N, D = local_weights_grad.shape
    
    # All-gather
    all_weights_grad = local_weights_grad
    all_ids = local_ids

    if world_size > 1:
        all_weights_grad = torch.empty((world_size * N, D), dtype=local_weights_grad.dtype, device=local_weights_grad.device)
        all_ids = torch.empty(world_size * N,               dtype=local_ids.dtype,          device=local_ids.device)
    
        dist.all_gather_into_tensor(all_weights_grad, local_weights_grad)
        dist.all_gather_into_tensor(all_ids,          local_ids)

    # Unique
    grad_ids, inv = all_ids.unique(return_inverse=True)

    grad = torch.zeros((grad_ids.shape[0], D), dtype=all_weights_grad.dtype, device=all_weights_grad.device)
    grad.scatter_add_(0, inv.unsqueeze(-1).expand(-1, D), all_weights_grad)

    # SignSGD with decoupled weight decay
    update_dtype = local_weights_grad.dtype
    grad = grad.to(update_dtype)
    p = weights[grad_ids].to(update_dtype)

    p.mul_(1.0 - lr * weight_decay).add_(torch.sign(grad), alpha=-lr)

    # Write updated slices back
    weights[grad_ids] = p.to(weights.dtype)
