"""
loss_weighting.py — Multi-Task Loss Weighting via Homoscedastic Uncertainty.

Owner:   Student A
Phase:   3 — Modeling & Training (Week 3–5)

Method:
    Kendall et al. (2018) "Multi-Task Learning Using Uncertainty to Weigh
    Losses in Scene Understanding." CVPR 2018.

    Instead of manually tuning λ_ner, λ_pos, λ_coref, the model learns
    optimal task weights via log-variance parameters (log σ²) per task.

    Loss for task i:
        L_i_weighted = (1 / 2σ_i²) * L_i + log(σ_i)

    The model minimizes total loss — large σ_i reduces L_i's contribution
    but is penalized by the log(σ_i) regularization term. Equilibrium gives
    the optimal weighting automatically.

Usage:
    weighter = UncertaintyWeighter(tasks=["ner", "pos", "coref"])
    optimizer includes weighter.parameters()

    total_loss, info = weighter(ner_loss=l1, pos_loss=l2, coref_loss=l3)
    total_loss.backward()
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Uncertainty-based loss weighter (Kendall et al. 2018)
# ---------------------------------------------------------------------------
class UncertaintyWeighter(nn.Module):
    """
    Learnable multi-task loss weighting via homoscedastic uncertainty.

    Args:
        tasks:       List of task names — order must match forward() kwargs.
        init_sigma:  Initial σ value per task (default 1.0 = equal weighting).
                     Use smaller values (e.g. 0.5) to give a task more weight
                     at the start of training.

    The learnable parameters are log(σ²) per task — unconstrained reals,
    which keeps gradients well-behaved (no clipping needed).
    """

    def __init__(
        self,
        tasks: list[str] = ("ner", "pos", "coref"),
        init_sigma: float = 1.0,
    ) -> None:
        super().__init__()
        self.tasks = list(tasks)
        n = len(tasks)

        # log(σ²) initialized from init_sigma
        init_log_var = math.log(init_sigma ** 2)
        self.log_vars = nn.Parameter(
            torch.full((n,), init_log_var, dtype=torch.float32)
        )

    # ── Properties ──────────────────────────────────────────────────────
    @property
    def sigmas(self) -> torch.Tensor:
        """σ per task — for logging / visualization."""
        return torch.exp(0.5 * self.log_vars)

    @property
    def weights(self) -> torch.Tensor:
        """Effective weight per task = 1 / (2σ²)."""
        return 0.5 * torch.exp(-self.log_vars)

    # ── Forward ─────────────────────────────────────────────────────────
    def forward(self, **task_losses: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Args:
            **task_losses:  Keyword losses matching self.tasks.
                            Missing tasks are skipped gracefully.
                            e.g. weighter(ner_loss=l1, pos_loss=l2)

        Returns:
            (total_loss, info_dict)

            info_dict keys:
              "total"           — scalar total loss
              "weighted_{task}" — weighted contribution of each task
              "sigma_{task}"    — current σ for each task (for W&B logging)
              "weight_{task}"   — effective weight 1/(2σ²) per task
        """
        total = torch.tensor(0.0, device=self.log_vars.device,
                             requires_grad=True)
        info: dict[str, float] = {}

        for i, task in enumerate(self.tasks):
            key = f"{task}_loss"
            if key not in task_losses:
                continue

            loss_i = task_losses[key]
            log_var_i = self.log_vars[i]

            # Kendall formula: L_weighted = (1/2σ²) * L + log(σ)
            #                             = exp(-log_var) * L + 0.5 * log_var
            weighted = torch.exp(-log_var_i) * loss_i + 0.5 * log_var_i
            total = total + weighted

            info[f"weighted_{task}"] = weighted.item()
            info[f"sigma_{task}"]    = self.sigmas[i].item()
            info[f"weight_{task}"]   = self.weights[i].item()

        info["total"] = total.item()
        return total, info


# ---------------------------------------------------------------------------
# Fixed-weight baseline (for ablation study)
# ---------------------------------------------------------------------------
class FixedWeighter(nn.Module):
    """
    Simple fixed-weight multi-task loss combiner.
    Use this as the ablation baseline against UncertaintyWeighter.

    Args:
        weights: dict mapping task name → float weight.
                 e.g. {"ner": 1.0, "pos": 0.5, "coref": 1.0}
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        super().__init__()
        self.weights = weights or {"ner": 1.0, "pos": 1.0, "coref": 1.0}

    def forward(self, **task_losses: torch.Tensor) -> tuple[torch.Tensor, dict]:
        total = torch.tensor(0.0)
        info: dict[str, float] = {}

        for task, w in self.weights.items():
            key = f"{task}_loss"
            if key not in task_losses:
                continue
            loss_i = task_losses[key]
            if total.device != loss_i.device:
                total = total.to(loss_i.device)
            weighted = w * loss_i
            total = total + weighted
            info[f"weighted_{task}"] = weighted.item()
            info[f"weight_{task}"]   = w

        info["total"] = total.item()
        return total, info


# ---------------------------------------------------------------------------
# Sanity check  (run: python loss_weighting.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Loss Weighting Sanity Check ===\n")

    # ── UncertaintyWeighter ──────────────────────────────────────────────
    weighter = UncertaintyWeighter(tasks=["ner", "pos", "coref"])

    ner_loss  = torch.tensor(2.5)
    pos_loss  = torch.tensor(0.8)
    coref_loss = torch.tensor(3.1)

    total, info = weighter(
        ner_loss=ner_loss,
        pos_loss=pos_loss,
        coref_loss=coref_loss,
    )

    assert total.requires_grad, "total_loss must have grad"
    print(f"✓ UncertaintyWeighter total loss: {total.item():.4f}")
    for k, v in info.items():
        print(f"    {k}: {v:.4f}")

    # ── Backward pass ────────────────────────────────────────────────────
    total.backward()
    assert weighter.log_vars.grad is not None
    print(f"\n✓ Backward pass OK — log_vars.grad: {weighter.log_vars.grad.tolist()}")

    # ── FixedWeighter (ablation baseline) ────────────────────────────────
    fixed = FixedWeighter({"ner": 1.0, "pos": 0.5, "coref": 1.0})
    total_f, info_f = fixed(
        ner_loss=ner_loss,
        pos_loss=pos_loss,
        coref_loss=coref_loss,
    )
    print(f"\n✓ FixedWeighter total loss: {total_f.item():.4f}")

    # ── Missing task graceful skip ────────────────────────────────────────
    total_skip, _ = weighter(ner_loss=ner_loss)  # pos + coref missing
    print(f"✓ Graceful skip (only NER): total={total_skip.item():.4f}")

    print("\n✅ All checks passed — place in src/training/loss_weighting.py")