"""
Loss functions for Bayesian Hyperbolic Zero-Shot Learning.

Three components are combined:

1. **Hyperbolic alignment loss** — pulls the sampled visual embedding z toward
   the correct class's semantic embedding on the Poincaré ball, while pushing
   it away from negative classes.  Operates on geodesic distances.

2. **KL regulariser** — penalises the wrapped-normal posterior
   q(z|x) = WN(μ, σ²) against a prior p(z) = WN(0, I) centred at the
   origin.  A closed-form approximation in the tangent space is used
   (Mathieu et al., 2019).

3. **Hierarchy-aware margin** — adjusts the contrastive margin per negative
   class proportional to its WordNet distance from the positive class,
   encouraging the geometry to respect the semantic tree.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt


# ---------------------------------------------------------------------------
# Hyperbolic contrastive alignment loss
# ---------------------------------------------------------------------------

class HyperbolicAlignmentLoss(nn.Module):
    """Contrastive ranking loss in Poincaré space.

    For each sample we compute geodesic distances to every class prototype
    and apply a cross-entropy-style softmax over negative distances.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z: torch.Tensor,
        class_embeddings: torch.Tensor,
        labels: torch.Tensor,
        ball: geoopt.PoincareBall,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z : (B, d)  sampled visual embeddings on the Poincaré ball.
        class_embeddings : (C, d)  all class prototypes on the ball.
        labels : (B,)  ground-truth class indices (into class_embeddings).
        ball : geoopt.PoincareBall  shared manifold.

        Returns
        -------
        Scalar cross-entropy-style loss.
        """
        # Pairwise geodesic distances  (B, C)
        # z: (B, d), class_embeddings: (C, d) → broadcast via unsqueeze
        dists = ball.dist(
            z.unsqueeze(1),               # (B, 1, d)
            class_embeddings.unsqueeze(0), # (1, C, d)
        )  # (B, C)

        # Convert distances to logits (closer = higher logit)
        logits = -dists / self.temperature  # (B, C)
        loss = F.cross_entropy(logits, labels)
        return loss


# ---------------------------------------------------------------------------
# Tangent-space KL divergence  (approx. wrapped-normal KL)
# ---------------------------------------------------------------------------

class WrappedNormalKL(nn.Module):
    """KL(q ‖ p) where q = WN(μ, diag(σ²)) and p = WN(0, I).

    Approximation: work in the tangent space at the origin. The KL reduces to
    the standard Gaussian KL plus a curvature-dependent volume correction.
    """

    def __init__(self, free_nats: float = 0.0):
        super().__init__()
        self.free_nats = free_nats  # per-dimension free-nats budget

    def forward(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        ball: geoopt.PoincareBall,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        mu : (B, d)  mean on the Poincaré ball.
        logvar : (B, d)  log-variance (Euclidean parameterisation).
        ball : geoopt.PoincareBall

        Returns
        -------
        Scalar KL divergence averaged over the batch.
        """
        # Map μ back to tangent space at origin for the Gaussian KL
        mu_tan = ball.logmap0(mu)  # (B, d)

        # Standard Gaussian KL: 0.5 * Σ(σ² + μ² − 1 − log σ²)
        kl = 0.5 * (logvar.exp() + mu_tan.pow(2) - 1.0 - logvar)  # (B, d)

        # Apply free-nats threshold per dimension
        if self.free_nats > 0.0:
            kl = kl.clamp(min=self.free_nats)

        return kl.sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# Hierarchy-aware margin  (optional)
# ---------------------------------------------------------------------------

def build_hierarchy_distances(
    n_classes: int,
    seen_classes: Optional[list] = None,
) -> torch.Tensor:
    """Placeholder: returns a (C, C) matrix of pairwise WordNet path-length
    distances between AwA2 classes.

    TODO: compute from nltk.corpus.wordnet once the class→synset mapping is
    finalised.  For now returns uniform distances (margin = 1).
    """
    dists = torch.ones(n_classes, n_classes)
    dists.fill_diagonal_(0.0)
    return dists


class HierarchyMarginLoss(nn.Module):
    """Contrastive loss with per-negative margin scaled by WordNet distance.

    Closer classes in the hierarchy get a *smaller* margin — the model is
    penalised less for confusing a leopard with a tiger than with a dolphin.
    """

    def __init__(
        self,
        hierarchy_dists: torch.Tensor,
        base_margin: float = 1.0,
        scale: float = 0.1,
    ):
        super().__init__()
        self.register_buffer("hierarchy_dists", hierarchy_dists)
        self.base_margin = base_margin
        self.scale = scale

    def forward(
        self,
        z: torch.Tensor,
        class_embeddings: torch.Tensor,
        labels: torch.Tensor,
        ball: geoopt.PoincareBall,
    ) -> torch.Tensor:
        """Hierarchy-weighted hinge loss."""
        B, d = z.shape
        C = class_embeddings.shape[0]

        dists = ball.dist(z.unsqueeze(1), class_embeddings.unsqueeze(0))  # (B, C)

        # Positive distances
        pos_dists = dists[torch.arange(B, device=z.device), labels]  # (B,)

        # Margin per negative  (B, C)
        margins = self.base_margin + self.scale * self.hierarchy_dists[labels]
        margins = margins.to(z.device)

        # Hinge: max(0,  d_pos − d_neg + margin)
        losses = F.relu(pos_dists.unsqueeze(1) - dists + margins)  # (B, C)

        # Zero out the positive position
        mask = torch.ones(B, C, device=z.device).bool()
        mask[torch.arange(B, device=z.device), labels] = False
        losses = losses[mask].view(B, C - 1)

        return losses.mean()


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

class BayesianHyperbolicLoss(nn.Module):
    """Weighted sum of alignment + KL + hierarchy losses.

    L = λ_align * L_align  +  λ_kl * L_KL  +  λ_hier * L_hierarchy
    """

    def __init__(
        self,
        temperature: float = 0.1,
        lambda_kl: float = 0.01,
        lambda_hier: float = 0.0,
        free_nats: float = 0.0,
        hierarchy_dists: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.alignment = HyperbolicAlignmentLoss(temperature)
        self.kl = WrappedNormalKL(free_nats)
        self.lambda_kl = lambda_kl
        self.lambda_hier = lambda_hier

        if hierarchy_dists is not None and lambda_hier > 0:
            self.hierarchy = HierarchyMarginLoss(hierarchy_dists)
        else:
            self.hierarchy = None

    def forward(
        self,
        model_out: dict,
        class_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict:
        """
        Parameters
        ----------
        model_out : dict from BayesianHyperbolicZSL.forward()
        class_embeddings : (C, d) class prototypes on the ball.
        labels : (B,) class indices.

        Returns
        -------
        dict with keys: 'total', 'align', 'kl', 'hier'
        """
        ball = model_out["ball"]
        z = model_out["z"]
        mu = model_out["mu"]
        logvar = model_out["logvar"]

        align = self.alignment(z, class_embeddings, labels, ball)
        kl = self.kl(mu, logvar, ball)

        total = align + self.lambda_kl * kl

        hier = torch.tensor(0.0, device=z.device)
        if self.hierarchy is not None:
            hier = self.hierarchy(z, class_embeddings, labels, ball)
            total = total + self.lambda_hier * hier

        return {"total": total, "align": align, "kl": kl, "hier": hier}


# ---------------------------------------------------------------------------
# Euclidean baseline loss
# ---------------------------------------------------------------------------

class EuclideanAlignmentLoss(nn.Module):
    """Cross-entropy over negative Euclidean distances — baseline."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        visual_emb: torch.Tensor,
        class_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        dists = torch.cdist(visual_emb, class_embeddings)  # (B, C)
        logits = -dists / self.temperature
        return F.cross_entropy(logits, labels)
