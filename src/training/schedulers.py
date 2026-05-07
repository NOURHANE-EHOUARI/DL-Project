"""
src/training/schedulers.py — Learning Rate Schedulers for Arabic MTL Training
==============================================================================
Owner  : Student A
Phase  : 3 — Modeling & Training (Week 3–5)

Implements the two-phase LR schedule used in the blueprint:
  1. Linear warmup   — ramp from 0 to peak LR over `warmup_steps`
  2. Cosine decay    — smooth decay from peak LR to `min_lr` over remaining steps

Why this schedule?
------------------
  - Warmup prevents large gradient updates early in fine-tuning when the
    task heads are randomly initialized (they would otherwise destabilize
    the pre-trained AraBERT weights).
  - Cosine decay is smoother than step decay and consistently outperforms
    linear decay on NLP fine-tuning tasks (Loshchilov & Hutter, ICLR 2017).
  - For MTL, a single scheduler drives the shared backbone LR while
    task-specific heads can optionally use a multiplier (see MTLScheduler).

Public API
----------
    # Standard use (wraps any PyTorch optimizer):
    scheduler = get_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=500,
        total_steps=10_000,
        min_lr_ratio=0.05,
    )
    scheduler.step()   # call once per training step

    # MTL-aware scheduler (per-head LR multipliers):
    scheduler = MTLScheduler(
        optimizer,
        warmup_steps=500,
        total_steps=10_000,
        head_lr_multipliers={"ner": 2.0, "pos": 2.0, "coref": 3.0},
    )
    scheduler.step()

    # Inspect current LR:
    current_lr = scheduler.get_last_lr()[0]
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core schedule function
# ─────────────────────────────────────────────────────────────────────────────

def _warmup_cosine_lambda(
    current_step: int,
    warmup_steps: int,
    total_steps:  int,
    min_lr_ratio: float = 0.0,
    num_cycles:   float = 0.5,
) -> float:
    """
    Compute the LR multiplier for a given step.

    Phase 1 — Warmup (0 ≤ step < warmup_steps):
        multiplier = step / warmup_steps
        Ramps linearly from 0.0 to 1.0.

    Phase 2 — Cosine decay (warmup_steps ≤ step ≤ total_steps):
        multiplier = min_lr_ratio + (1 - min_lr_ratio) *
                     0.5 * (1 + cos(π * progress))
        where progress = (step - warmup_steps) / (total_steps - warmup_steps)

        At step = warmup_steps  → multiplier = 1.0  (peak)
        At step = total_steps   → multiplier = min_lr_ratio

    Args:
        current_step: Current training step (0-indexed).
        warmup_steps: Number of warmup steps.
        total_steps:  Total number of training steps.
        min_lr_ratio: Minimum LR as a fraction of peak LR. Default 0.0
                      means LR decays to 0. Set to 0.05 for a non-zero floor.
        num_cycles:   Number of cosine half-cycles. Default 0.5 = one decay.
                      Use 1.0 for cosine annealing with restart.

    Returns:
        Float multiplier in [min_lr_ratio, 1.0].
    """
    # Clamp to valid range
    current_step = min(current_step, total_steps)

    if current_step < warmup_steps:
        # Linear warmup
        return float(current_step) / float(max(1, warmup_steps))

    # Cosine decay
    progress = float(current_step - warmup_steps) / float(
        max(1, total_steps - warmup_steps)
    )
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay


# ─────────────────────────────────────────────────────────────────────────────
# Factory function — drop-in for HuggingFace get_scheduler
# ─────────────────────────────────────────────────────────────────────────────

def get_warmup_cosine_scheduler(
    optimizer:    Optimizer,
    warmup_steps: int,
    total_steps:  int,
    min_lr_ratio: float = 0.0,
    num_cycles:   float = 0.5,
    last_epoch:   int   = -1,
) -> LambdaLR:
    """
    Build a warmup + cosine decay LambdaLR scheduler.

    This is the recommended scheduler for AraBERT fine-tuning.
    Compatible with any PyTorch optimizer and Hugging Face Trainer.

    Args:
        optimizer:    PyTorch optimizer (AdamW recommended).
        warmup_steps: Steps for linear warmup (typically 5–10% of total).
        total_steps:  Total training steps = epochs × steps_per_epoch.
        min_lr_ratio: LR floor as fraction of peak. 0.05 is a safe default.
        num_cycles:   Cosine half-cycles. Default 0.5 = one smooth decay.
        last_epoch:   Resume from this step (-1 = start fresh).

    Returns:
        LambdaLR scheduler. Call `.step()` once per training step.

    Example:
        optimizer = AdamW(model.parameters(), lr=2e-5)
        scheduler = get_warmup_cosine_scheduler(
            optimizer,
            warmup_steps=500,
            total_steps=10_000,
            min_lr_ratio=0.05,
        )
        for step, batch in enumerate(dataloader):
            loss = model(batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    """
    if warmup_steps >= total_steps:
        logger.warning(
            "warmup_steps (%d) >= total_steps (%d). "
            "Warmup will cover the entire training run.",
            warmup_steps, total_steps,
        )

    lambda_fn = lambda step: _warmup_cosine_lambda(   # noqa: E731
        step, warmup_steps, total_steps, min_lr_ratio, num_cycles
    )
    return LambdaLR(optimizer, lr_lambda=lambda_fn, last_epoch=last_epoch)


# ─────────────────────────────────────────────────────────────────────────────
# Warmup-only scheduler (for ablation: warmup without cosine)
# ─────────────────────────────────────────────────────────────────────────────

def get_linear_warmup_scheduler(
    optimizer:    Optimizer,
    warmup_steps: int,
    total_steps:  int,
    last_epoch:   int = -1,
) -> LambdaLR:
    """
    Linear warmup then constant LR.

    Used as an ablation condition: same warmup, no cosine decay.
    Lets us isolate the effect of cosine decay in the ablation study.
    """
    def lambda_fn(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda=lambda_fn, last_epoch=last_epoch)


def get_constant_scheduler(
    optimizer:  Optimizer,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Constant LR (no warmup, no decay). Ablation baseline.
    """
    return LambdaLR(optimizer, lr_lambda=lambda _: 1.0, last_epoch=last_epoch)


# ─────────────────────────────────────────────────────────────────────────────
# MTL-aware scheduler — per-head LR multipliers
# ─────────────────────────────────────────────────────────────────────────────

class MTLScheduler:
    """
    Multi-Task Learning scheduler with per-parameter-group LR multipliers.

    In MTL fine-tuning, task heads are randomly initialized and need a
    higher LR than the pre-trained backbone. A typical configuration:
        backbone LR  = 2e-5   (base)
        head LR      = 4e-5   (2× multiplier)
        coref LR     = 6e-5   (3× multiplier — coref head is more complex)

    This is achieved by passing multiple parameter groups to the optimizer:
        optimizer = AdamW([
            {"params": backbone.parameters(), "lr": 2e-5},
            {"params": ner_head.parameters(), "lr": 2e-5, "group": "ner"},
            {"params": pos_head.parameters(), "lr": 2e-5, "group": "pos"},
            {"params": coref_head.parameters(), "lr": 2e-5, "group": "coref"},
        ])

    MTLScheduler applies the same warmup+cosine schedule to all groups,
    then multiplies each group's LR by its head multiplier.

    Args:
        optimizer:           PyTorch optimizer with named parameter groups.
        warmup_steps:        Warmup steps (same for all groups).
        total_steps:         Total training steps.
        min_lr_ratio:        LR floor (same for all groups).
        head_lr_multipliers: Dict mapping param group name → multiplier.
                             Groups not listed use multiplier 1.0.
        group_key:           Key in optimizer param_groups used to identify
                             the group name. Default "group".

    Example:
        optimizer = AdamW([
            {"params": model.backbone.parameters(), "lr": 2e-5},
            {"params": model.ner_head.parameters(),   "lr": 2e-5, "group": "ner"},
            {"params": model.pos_head.parameters(),   "lr": 2e-5, "group": "pos"},
            {"params": model.coref_head.parameters(), "lr": 2e-5, "group": "coref"},
        ], weight_decay=0.01)

        scheduler = MTLScheduler(
            optimizer,
            warmup_steps=500,
            total_steps=10_000,
            head_lr_multipliers={"ner": 2.0, "pos": 2.0, "coref": 3.0},
        )
        # In training loop:
        scheduler.step()
    """

    def __init__(
        self,
        optimizer:           Optimizer,
        warmup_steps:        int,
        total_steps:         int,
        min_lr_ratio:        float = 0.0,
        num_cycles:          float = 0.5,
        head_lr_multipliers: Optional[dict[str, float]] = None,
        group_key:           str = "group",
        last_epoch:          int = -1,
    ):
        self.optimizer           = optimizer
        self.warmup_steps        = warmup_steps
        self.total_steps         = total_steps
        self.min_lr_ratio        = min_lr_ratio
        self.num_cycles          = num_cycles
        self.head_lr_multipliers = head_lr_multipliers or {}
        self.group_key           = group_key
        self._step_count         = max(0, last_epoch)

        # Store the base LR for each param group (set at optimizer creation)
        self._base_lrs: list[float] = [
            pg["lr"] for pg in optimizer.param_groups
        ]

        logger.info(
            "MTLScheduler initialized: warmup=%d, total=%d, "
            "multipliers=%s",
            warmup_steps, total_steps, self.head_lr_multipliers,
        )

    def step(self) -> None:
        """Advance the scheduler by one step and update optimizer LRs."""
        multiplier = _warmup_cosine_lambda(
            self._step_count,
            self.warmup_steps,
            self.total_steps,
            self.min_lr_ratio,
            self.num_cycles,
        )
        self._step_count += 1

        for pg, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
            group_name  = pg.get(self.group_key, None)
            head_mult   = self.head_lr_multipliers.get(group_name, 1.0)
            pg["lr"]    = base_lr * multiplier * head_mult

    def get_last_lr(self) -> list[float]:
        """Return the LR of each parameter group after the last step."""
        return [pg["lr"] for pg in self.optimizer.param_groups]

    def get_lr_for_step(self, step: int) -> float:
        """Return the base LR multiplier for a given step (no side effects)."""
        return _warmup_cosine_lambda(
            step, self.warmup_steps, self.total_steps,
            self.min_lr_ratio, self.num_cycles,
        )

    def state_dict(self) -> dict[str, Any]:
        """Serialize scheduler state for checkpointing."""
        return {
            "warmup_steps":        self.warmup_steps,
            "total_steps":         self.total_steps,
            "min_lr_ratio":        self.min_lr_ratio,
            "num_cycles":          self.num_cycles,
            "head_lr_multipliers": self.head_lr_multipliers,
            "group_key":           self.group_key,
            "_step_count":         self._step_count,
            "_base_lrs":           self._base_lrs,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore scheduler state from checkpoint."""
        self.warmup_steps        = state["warmup_steps"]
        self.total_steps         = state["total_steps"]
        self.min_lr_ratio        = state["min_lr_ratio"]
        self.num_cycles          = state["num_cycles"]
        self.head_lr_multipliers = state["head_lr_multipliers"]
        self.group_key           = state["group_key"]
        self._step_count         = state["_step_count"]
        self._base_lrs           = state["_base_lrs"]
        logger.info("MTLScheduler state restored at step %d", self._step_count)

    def __repr__(self) -> str:
        return (
            f"MTLScheduler("
            f"warmup={self.warmup_steps}, "
            f"total={self.total_steps}, "
            f"step={self._step_count}, "
            f"multipliers={self.head_lr_multipliers})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Schedule builder — used by trainer.py
# ─────────────────────────────────────────────────────────────────────────────

def build_scheduler(
    name:         str,
    optimizer:    Optimizer,
    warmup_steps: int,
    total_steps:  int,
    min_lr_ratio: float = 0.05,
    **kwargs:     Any,
) -> LambdaLR | MTLScheduler:
    """
    Build a scheduler by name. Used by trainer.py to avoid hard-coding.

    Args:
        name:         One of 'warmup_cosine' (default), 'linear_warmup',
                      'constant', 'mtl'.
        optimizer:    PyTorch optimizer.
        warmup_steps: Warmup steps.
        total_steps:  Total training steps.
        min_lr_ratio: LR floor for cosine schedules.
        **kwargs:     Passed to the scheduler constructor
                      (e.g. head_lr_multipliers for 'mtl').

    Returns:
        Scheduler instance.

    Raises:
        ValueError: If name is not recognized.
    """
    name = name.lower().strip()

    if name == "warmup_cosine":
        return get_warmup_cosine_scheduler(
            optimizer, warmup_steps, total_steps,
            min_lr_ratio=min_lr_ratio,
            num_cycles=kwargs.get("num_cycles", 0.5),
        )
    elif name == "linear_warmup":
        return get_linear_warmup_scheduler(
            optimizer, warmup_steps, total_steps,
        )
    elif name == "constant":
        return get_constant_scheduler(optimizer)
    elif name == "mtl":
        return MTLScheduler(
            optimizer, warmup_steps, total_steps,
            min_lr_ratio=min_lr_ratio,
            head_lr_multipliers=kwargs.get("head_lr_multipliers", {}),
        )
    else:
        raise ValueError(
            f"Unknown scheduler '{name}'. "
            "Choose from: warmup_cosine, linear_warmup, constant, mtl."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def compute_total_steps(
    num_train_samples: int,
    batch_size:        int,
    num_epochs:        int,
    gradient_accumulation_steps: int = 1,
) -> int:
    """
    Compute total training steps from dataset size and training config.

    Formula:
        steps_per_epoch = ceil(num_train_samples / batch_size)
        total_steps     = steps_per_epoch × num_epochs
                          / gradient_accumulation_steps

    Used in trainer.py to avoid computing this inline.

    Example:
        total = compute_total_steps(
            num_train_samples=3525,   # ANERcorp train
            batch_size=32,
            num_epochs=10,
            gradient_accumulation_steps=2,
        )
        # → 555 optimizer steps
    """
    steps_per_epoch = math.ceil(num_train_samples / batch_size)
    total_steps     = steps_per_epoch * num_epochs
    optimizer_steps = math.ceil(total_steps / gradient_accumulation_steps)
    return optimizer_steps


def compute_warmup_steps(
    total_steps:   int,
    warmup_ratio:  float = 0.06,
) -> int:
    """
    Compute warmup steps as a fraction of total steps.

    The standard for BERT fine-tuning is 6% warmup (Devlin et al., 2019).
    For Arabic MTL with smaller datasets, 10% is safer.

    Args:
        total_steps:  Total training steps.
        warmup_ratio: Fraction of total steps for warmup. Default 0.06.

    Returns:
        Number of warmup steps (minimum 1).
    """
    return max(1, int(total_steps * warmup_ratio))


def get_schedule_preview(
    warmup_steps: int,
    total_steps:  int,
    min_lr_ratio: float = 0.0,
    peak_lr:      float = 2e-5,
    n_points:     int   = 20,
) -> list[dict[str, float]]:
    """
    Generate a preview of the LR schedule for visualization / logging.

    Returns a list of {step, lr} dicts evenly sampled across the schedule.
    Useful for W&B logging at the start of training:

        preview = get_schedule_preview(warmup_steps=500, total_steps=10_000)
        wandb.log({"lr_schedule": wandb.plot.line_series(...)})
    """
    checkpoints = [
        int(i * total_steps / (n_points - 1)) for i in range(n_points)
    ]
    return [
        {
            "step": step,
            "lr":   peak_lr * _warmup_cosine_lambda(
                step, warmup_steps, total_steps, min_lr_ratio
            ),
            "multiplier": _warmup_cosine_lambda(
                step, warmup_steps, total_steps, min_lr_ratio
            ),
        }
        for step in checkpoints
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch
    from torch.optim import AdamW

    print("\n── Warmup + Cosine schedule preview ────────────────────")
    warmup, total = 500, 5000
    preview = get_schedule_preview(warmup, total, min_lr_ratio=0.05,
                                   peak_lr=2e-5, n_points=10)
    for p in preview:
        bar = "█" * int(p["multiplier"] * 30)
        print(f"  step {p['step']:5d}  |  {bar:<30s}  lr={p['lr']:.2e}")

    print("\n── MTLScheduler test ────────────────────────────────────")
    # Dummy model with 3 param groups
    backbone = torch.nn.Linear(768, 768)
    ner_head = torch.nn.Linear(768, 27)
    pos_head = torch.nn.Linear(768, 35)

    opt = AdamW([
        {"params": backbone.parameters(), "lr": 2e-5},
        {"params": ner_head.parameters(), "lr": 2e-5, "group": "ner"},
        {"params": pos_head.parameters(), "lr": 2e-5, "group": "pos"},
    ], weight_decay=0.01)

    scheduler = MTLScheduler(
        opt,
        warmup_steps=10,
        total_steps=100,
        head_lr_multipliers={"ner": 2.0, "pos": 2.0},
        min_lr_ratio=0.05,
    )

    print(f"\n  {'Step':>6}  {'backbone LR':>14}  {'NER LR':>12}  {'POS LR':>12}")
    print(f"  {'────':>6}  {'───────────':>14}  {'──────':>12}  {'──────':>12}")
    for step in [0, 5, 10, 25, 50, 75, 99]:
        scheduler._step_count = step
        scheduler.step()
        lrs = scheduler.get_last_lr()
        print(f"  {step:6d}  {lrs[0]:>14.2e}  {lrs[1]:>12.2e}  {lrs[2]:>12.2e}")

    print("\n── compute_total_steps test ─────────────────────────────")
    steps = compute_total_steps(3525, batch_size=32, num_epochs=10,
                                gradient_accumulation_steps=2)
    warmup = compute_warmup_steps(steps, warmup_ratio=0.06)
    print(f"  ANERcorp (3525 samples): {steps} optimizer steps, "
          f"{warmup} warmup steps")

    print("\n── build_scheduler factory test ─────────────────────────")
    for name in ["warmup_cosine", "linear_warmup", "constant"]:
        s = build_scheduler(name, opt, warmup_steps=50, total_steps=500)
        print(f"  ✅ {name}")

    print("\n  schedulers.py ready.\n")