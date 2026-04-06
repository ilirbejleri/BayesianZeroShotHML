"""
Bayesian Hyperbolic Zero-Shot Learning — model architectures.

We implement a visual-semantic bridge that maps Euclidean visual features
(ResNet-101 or ViT) to *distributions* on the Poincaré ball rather than
deterministic points.  Concretely, the encoder outputs:

    μ  ∈ 𝔹ⁿ_c   (Poincaré mean via exponential map)
    σ  ∈ ℝ₊ⁿ    (per-dimension log-variance, clamped for stability)

Together (μ, σ) parameterise a Wrapped Normal distribution on the manifold
(Nagano et al., 2019; Mathieu et al., 2019).  Sampling is performed in the
tangent space at μ and pushed onto the manifold via the exponential map,
making the reparameterisation trick fully differentiable.

Curvature c is a *learnable* scalar following Ganea et al. (2018).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt

# ---------------------------------------------------------------------------
# Poincaré-ball helpers  (curvature-aware, following Ganea et al. 2018)
# ---------------------------------------------------------------------------

class PoincareOperations:
    """Stateless helper wrapping geoopt's PoincareBall with a learnable curvature."""

    def __init__(self, ball: geoopt.PoincareBall):
        self.ball = ball

    def expmap0(self, v: torch.Tensor) -> torch.Tensor:
        """Exponential map from the origin."""
        return self.ball.expmap0(v)

    def logmap0(self, y: torch.Tensor) -> torch.Tensor:
        """Logarithmic map to the origin."""
        return self.ball.logmap0(y)

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Poincaré distance."""
        return self.ball.dist(x, y)

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ball.mobius_add(x, y)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project onto the open ball (clamp norm < 1/√c − ε)."""
        return self.ball.projx(x)


# ---------------------------------------------------------------------------
# Hyperbolic linear layer  (Ganea et al. 2018 §4)
# ---------------------------------------------------------------------------

class HyperbolicLinear(nn.Module):
    """Möbius matrix–vector multiplication followed by Möbius bias addition.

    Operates in the tangent space at the origin, applies a standard linear map,
    then maps back via the exponential map — equivalent to the Möbius
    formulation but numerically more stable.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ball: geoopt.PoincareBall,
        bias: bool = True,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.ball = ball
        if bias:
            # Bias lives on the manifold
            self.bias = geoopt.ManifoldParameter(
                torch.zeros(out_features), manifold=ball,
            )
        else:
            self.bias = None

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.linear.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x assumed Euclidean (tangent space at origin) or already on manifold
        h = self.linear(x)
        h = self.ball.expmap0(h)
        if self.bias is not None:
            h = self.ball.mobius_add(h, self.bias)
        h = self.ball.projx(h)
        return h


# ---------------------------------------------------------------------------
# Wrapped Normal sampling (reparameterised)
# ---------------------------------------------------------------------------

def sample_wrapped_normal(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    ball: geoopt.PoincareBall,
    n_samples: int = 1,
) -> torch.Tensor:
    """Sample from a Wrapped Normal on the Poincaré ball.

    1. Sample ε ~ N(0, I) in ℝᵈ (tangent space at origin).
    2. Scale by σ = exp(0.5 * logvar).
    3. Parallel-transport the tangent vector from origin to μ.
    4. Apply the exponential map at μ to land on the manifold.

    Returns shape (B, d) when n_samples=1 or (n_samples, B, d).
    """
    std = (0.5 * logvar).exp()  # (B, d)
    if n_samples > 1:
        std = std.unsqueeze(0).expand(n_samples, -1, -1)
        mu_exp = mu.unsqueeze(0).expand(n_samples, -1, -1)
    else:
        mu_exp = mu

    eps = torch.randn_like(std)
    v = eps * std  # tangent vector at origin

    # Parallel transport from origin → μ, then expmap at μ
    v_transported = ball.transp0(mu_exp, v)
    z = ball.expmap(mu_exp, v_transported)
    z = ball.projx(z)

    if n_samples == 1:
        return z
    return z  # (n_samples, B, d)


# ---------------------------------------------------------------------------
# Visual encoder  (frozen backbone + projection head)
# ---------------------------------------------------------------------------

class VisualEncoder(nn.Module):
    """Projects pre-extracted visual features into the Poincaré ball.

    When ``from_features=True`` (default), we skip the backbone entirely and
    expect 2048-d ResNet-101 feature vectors as input.  Set
    ``from_features=False`` to attach a live (frozen) backbone for raw images.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 1024,
        embed_dim: int = 64,
        ball: Optional[geoopt.PoincareBall] = None,
    ):
        super().__init__()
        if ball is None:
            ball = geoopt.PoincareBall(c=1.0, learnable=True)
        self.ball = ball

        # Euclidean MLP head → tangent-space representation
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Outputs mean and log-variance for the wrapped normal
        self.mu_head = HyperbolicLinear(hidden_dim, embed_dim, ball)
        self.logvar_head = nn.Linear(hidden_dim, embed_dim)

    def forward(
        self, x: torch.Tensor, n_samples: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        mu : (B, d)  — mean on the Poincaré ball
        logvar : (B, d)  — log-variance (Euclidean)
        z : (B, d) or (n_samples, B, d)  — sampled embedding(s)
        """
        h = self.fc(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(-6.0, 2.0)  # stability
        z = sample_wrapped_normal(mu, logvar, self.ball, n_samples)
        return mu, logvar, z


# ---------------------------------------------------------------------------
# Semantic encoder  (attribute vector → Poincaré ball)
# ---------------------------------------------------------------------------

class SemanticEncoder(nn.Module):
    """Maps 85-d AwA2 attribute vectors to the same Poincaré ball."""

    def __init__(
        self,
        attr_dim: int = 85,
        hidden_dim: int = 256,
        embed_dim: int = 64,
        ball: Optional[geoopt.PoincareBall] = None,
    ):
        super().__init__()
        if ball is None:
            ball = geoopt.PoincareBall(c=1.0, learnable=True)
        self.ball = ball

        self.fc = nn.Sequential(
            nn.Linear(attr_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.hyp_proj = HyperbolicLinear(hidden_dim, embed_dim, ball)

    def forward(self, attr: torch.Tensor) -> torch.Tensor:
        """Returns deterministic point embedding on the Poincaré ball."""
        h = self.fc(attr)
        return self.hyp_proj(h)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class BayesianHyperbolicZSL(nn.Module):
    """End-to-end Bayesian Hyperbolic Zero-Shot Learning model.

    Combines the visual encoder (outputs a distribution) and the semantic
    encoder (outputs a point) on a shared Poincaré ball.
    """

    def __init__(
        self,
        visual_dim: int = 2048,
        attr_dim: int = 85,
        hidden_dim: int = 1024,
        embed_dim: int = 64,
        curvature: float = 1.0,
    ):
        super().__init__()
        self.ball = geoopt.PoincareBall(c=curvature, learnable=True)
        self.visual_encoder = VisualEncoder(
            input_dim=visual_dim,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            ball=self.ball,
        )
        self.semantic_encoder = SemanticEncoder(
            attr_dim=attr_dim,
            hidden_dim=256,
            embed_dim=embed_dim,
            ball=self.ball,
        )

    def forward(
        self,
        features: torch.Tensor,
        attributes: torch.Tensor,
        n_samples: int = 1,
    ) -> dict:
        mu, logvar, z = self.visual_encoder(features, n_samples)
        sem = self.semantic_encoder(attributes)
        return {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "semantic": sem,
            "ball": self.ball,
        }

    def embed_classes(self, class_attributes: torch.Tensor) -> torch.Tensor:
        """Embed all class attribute vectors — used at inference time."""
        return self.semantic_encoder(class_attributes)


# ---------------------------------------------------------------------------
# Euclidean baseline  (for ablation comparison)
# ---------------------------------------------------------------------------

class EuclideanBaselineZSL(nn.Module):
    """Deterministic Euclidean ZSL baseline for comparison."""

    def __init__(
        self,
        visual_dim: int = 2048,
        attr_dim: int = 85,
        hidden_dim: int = 1024,
        embed_dim: int = 64,
    ):
        super().__init__()
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.semantic_encoder = nn.Sequential(
            nn.Linear(attr_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embed_dim),
        )

    def forward(self, features: torch.Tensor, attributes: torch.Tensor) -> dict:
        vis = self.visual_encoder(features)
        sem = self.semantic_encoder(attributes)
        return {"visual": vis, "semantic": sem}

    def embed_classes(self, class_attributes: torch.Tensor) -> torch.Tensor:
        return self.semantic_encoder(class_attributes)
