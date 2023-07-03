"""DiCE: The Infinitely Differentiable Monte-Carlo Estimator"""

from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


def magic_box(tau: torch.Tensor) -> torch.Tensor:
    """MagicBox operator."""
    return torch.exp(tau - tau.detach())


def left_sum_to_size(tensor: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Sum the tensor to the given shape using left broadcasting."""
    return tensor.sum_to_size(shape + (1,) * (tensor.ndim - len(shape))).view(shape)


def cost_node(cost: torch.Tensor, logps: Iterable[torch.Tensor]) -> torch.Tensor:
    """Compute the surrogate losses for a set of stochastic nodes.

    Args:
        cost (torch.Tensor): Cost node (a scalar cost or a batch of costs).
        logps (Iterable[torch.Tensor]): Iterable of logprobs of stochastic nodes that the cost node
            causally depends on. The cost tensor must be left broadcastable to them.

    Returns:
        torch.Tensor: DiCE surrogate losses, the same shape as the cost tensor.
    """
    tau = sum((left_sum_to_size(logp, cost.shape) for logp in logps), cost.new_tensor(0.0))
    return magic_box(tau) * cost


def logp_categorical(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Compute the logprobs of optionally batched actions given logits.

    Args:
        logits (torch.Tensor): Logits of an optionally batched categorical distribution.
        actions (torch.Tensor): Actions to compute the log-probability of.

    Returns:
        torch.Tensor: Logprobs of the actions selected at this stochastic node.
    """
    return F.log_softmax(logits, dim=-1).gather(-1, actions[..., None])[..., 0]


def sample_categorical(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample from an optionally batched categorical distribution given logits.

    Args:
        logits (torch.Tensor): Logits of an optionally batched categorical distribution.

    Returns:
        torch.Tensor: Logprobs of the actions selected at this stochastic node.
        torch.Tensor: Sampled actions for this stochastic node.
    """
    g = torch.rand_like(logits).log_().neg_().log_().neg_()
    actions = torch.argmax(logits + g, dim=-1)
    logp = logp_categorical(logits, actions)
    return logp, actions


def baseline_term(
    baseline: torch.Tensor, logps: Iterable[torch.Tensor]
) -> torch.Tensor:
    """Compute the baseline term for a set of stochastic nodes.

    Args:
        baseline (torch.Tensor): Baseline. Any function of nodes not influenced by the stochastic
            nodes in `logps`.
        logps (Iterable[torch.Tensor]): Iterable of logprobs of stochastic nodes to compute
            the baseline term for.

    Returns:
        torch.Tensor: DiCE baseline term, a tensor the same shape as `baseline` that is zero in
            the forward pass and has the gradient of baseline subtraction.
    """
    logps = (left_sum_to_size(logp, baseline.shape) for logp in logps)
    terms = ((1 - magic_box(logp)) * baseline for logp in logps)
    return sum(terms, torch.tensor(0.0))


def batch_baseline_term(
    cost: torch.Tensor, logps: Iterable[torch.Tensor]
) -> torch.Tensor:
    """REINFORCE with replacement baseline, from "Buy 4 REINFORCE Samples, Get a Baseline for
    Free!". Computes a baseline for a batch of costs using the other costs in the batch.

    Args:
        cost (torch.Tensor): Batch of costs to compute the baseline for.
        logps (Iterable[torch.Tensor]): Iterable of logprobs of stochastic nodes to compute
            the baseline term for. The cost tensor must be left broadcastable to them.

    Returns:
        torch.Tensor: DiCE baseline term, a tensor the same shape as `cost` that is zero in
            the forward pass and has the gradient of baseline subtraction.
    """
    if cost.numel() <= 1:
        raise ValueError("batch_baseline_term() requires a batch of at least two costs")
    baseline = (cost.sum() - cost) / (cost.numel() - 1)
    logps = (left_sum_to_size(logp, baseline.shape) for logp in logps)
    terms = ((1 - magic_box(logp)) * baseline for logp in logps)
    return sum(terms, cost.new_tensor(0.0))


class EMABaseline(nn.Module):
    """Exponential moving average baseline.

    Args:
        decay (float): Decay rate. Defaults to 0.99.
    """

    def __init__(self, decay: float = 0.99):
        super().__init__()
        self.decay = decay
        self.register_buffer("mean_biased", torch.tensor(0.0))
        self.register_buffer("decay_cumprod", torch.tensor(1.0))

    def extra_repr(self) -> str:
        return f"decay={self.decay:g}"

    @property
    def mean(self) -> torch.Tensor:
        """The current mean cost."""
        return self.mean_biased / (1 - self.decay_cumprod)

    @torch.no_grad()
    def update(self, cost: torch.Tensor) -> None:
        """Update the baseline.

        Args:
            cost (torch.Tensor): Cost or batch of costs to update the baseline with.
        """
        self.decay_cumprod.mul_(self.decay)
        self.mean_biased.mul_(self.decay).add_(cost.mean(), alpha=1 - self.decay)

    def forward(
        self, cost: torch.Tensor, logps: Iterable[torch.Tensor]
    ) -> torch.Tensor:
        """Compute the baseline term, update the baseline, and return the cost with modified
        gradients.

        Args:
            cost (torch.Tensor): Cost or batch of costs to update the baseline with.
            logps (Iterable[torch.Tensor]): Iterable of logprobs of stochastic nodes to compute
                the baseline term for.

        Returns:
            torch.Tensor: DiCE baseline term, a tensor the same shape as `cost` that is zero in
                the forward pass and has the gradient of baseline subtraction.
        """
        if self.decay_cumprod < 1.0:
            baseline = baseline_term(self.mean.expand_as(cost), logps)
        else:
            baseline = torch.zeros_like(cost)
        self.update(cost)
        return baseline


__all__ = [
    magic_box,
    cost_node,
    logp_categorical,
    sample_categorical,
    baseline_term,
    batch_baseline_term,
    EMABaseline,
]
