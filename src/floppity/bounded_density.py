"""Bounded parameter transforms for SBI conditional density estimators."""

from typing import Tuple

import torch
from torch import Tensor
from sbi.neural_nets.estimators.base import ConditionalDensityEstimator


class ScaledLogitDensityEstimator(ConditionalDensityEstimator):
    """Expose an unconstrained estimator as a density on a box.

    The wrapped estimator learns ``z = logit((theta - low) / (high - low))``.
    Its public API still accepts and returns ``theta`` in natural units, including
    the change-of-variables Jacobian in every density evaluation.
    """

    def __init__(
        self,
        estimator: ConditionalDensityEstimator,
        low: Tensor,
        high: Tensor,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(
            estimator.net,
            input_shape=estimator.input_shape,
            condition_shape=estimator.condition_shape,
        )
        low = torch.as_tensor(low).reshape(estimator.input_shape)
        high = torch.as_tensor(high).reshape(estimator.input_shape)
        if not torch.all(torch.isfinite(low)) or not torch.all(torch.isfinite(high)):
            raise ValueError("Scaled-logit bounds must be finite.")
        if not torch.all(high > low):
            raise ValueError("Every scaled-logit upper bound must exceed its lower bound.")
        if not 0 < float(eps) < 0.5:
            raise ValueError("Scaled-logit eps must lie strictly between 0 and 0.5.")

        self.register_buffer("low", low)
        self.register_buffer("high", high)
        self.eps = float(eps)
        # The SBI wrapper contains no trainable state beyond ``net``. Keeping this
        # non-module reference lets us reuse its shape-aware API without registering
        # the same network twice in the PyTorch module tree.
        object.__setattr__(self, "_estimator", estimator)

    @property
    def embedding_net(self):
        return self._estimator.embedding_net

    @property
    def width(self) -> Tensor:
        return self.high - self.low

    def _unit_coordinates(self, theta: Tensor) -> Tensor:
        return (theta - self.low) / self.width

    def to_unconstrained(self, theta: Tensor) -> Tensor:
        """Map natural parameters to finite logit coordinates."""
        unit = self._unit_coordinates(theta).clamp(self.eps, 1.0 - self.eps)
        return torch.logit(unit)

    def to_natural(self, unconstrained: Tensor) -> Tensor:
        """Map logit coordinates into the open prior box."""
        unit = torch.sigmoid(unconstrained)
        return self.low + self.width * unit

    def _log_abs_det_jacobian(self, theta: Tensor) -> Tensor:
        unit = self._unit_coordinates(theta).clamp(self.eps, 1.0 - self.eps)
        per_dimension = (
            -torch.log(self.width)
            - torch.log(unit)
            - torch.log1p(-unit)
        )
        event_dims = tuple(
            range(per_dimension.ndim - len(self.input_shape), per_dimension.ndim)
        )
        return per_dimension.sum(dim=event_dims)

    def log_prob(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        inside = ((input >= self.low) & (input <= self.high)).all(dim=-1)
        unconstrained = self.to_unconstrained(input)
        log_prob = self._estimator.log_prob(unconstrained, condition, **kwargs)
        log_prob = log_prob + self._log_abs_det_jacobian(input)
        return torch.where(inside, log_prob, torch.full_like(log_prob, -torch.inf))

    def loss(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        unconstrained = self.to_unconstrained(input)
        return self._estimator.loss(
            unconstrained, condition, **kwargs
        ) - self._log_abs_det_jacobian(input)

    def sample(self, sample_shape: torch.Size, condition: Tensor, **kwargs) -> Tensor:
        unconstrained = self._estimator.sample(sample_shape, condition, **kwargs)
        return self.to_natural(unconstrained)

    def sample_and_log_prob(
        self, sample_shape: torch.Size, condition: Tensor, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        samples = self.sample(sample_shape, condition, **kwargs)
        return samples, self.log_prob(samples, condition, **kwargs)

    def inverse_transform(self, input: Tensor, condition: Tensor) -> Tensor:
        """Map natural parameters through the full flow into base-noise space."""
        return self._estimator.inverse_transform(
            self.to_unconstrained(input), condition
        )


def scaled_logit_builder(base_builder, low: Tensor, high: Tensor, eps: float = 1e-6):
    """Wrap an SBI density-estimator builder with a scaled-logit parameter map."""
    low = torch.as_tensor(low).detach().clone()
    high = torch.as_tensor(high).detach().clone()

    def build(theta: Tensor, x: Tensor):
        unit = ((theta - low.to(theta)) / (high.to(theta) - low.to(theta))).clamp(
            eps, 1.0 - eps
        )
        estimator = base_builder(torch.logit(unit), x)
        return ScaledLogitDensityEstimator(estimator, low=low, high=high, eps=eps)

    return build
