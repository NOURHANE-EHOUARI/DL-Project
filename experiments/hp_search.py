"""
hp_search.py — Hyperparameter Search via Optuna + W&B.

Owner:   Student A
Phase:   3 — Modeling & Training (Week 3–5)

What it tunes:
    learning_rate, batch_size, dropout, lstm_hidden,
    backbone_lr_multiplier, warmup_ratio, loss init_sigma

How to run:
    python experiments/hp_search.py --n_trials 30 --study_name arabert_ner

Requirements:
    pip install optuna optuna-integration[wandb] pyyaml
"""

from __future__ import annotations

import argparse
import os
import yaml
import optuna
from pathlib import Path

# ── Optuna logging setup ──────────────────────────────────────────────────
optuna.logging.set_verbosity(optuna.logging.INFO)

BASE_CONFIG_PATH = Path(__file__).parent / "configs" / "base.yaml"


def load_base_config() -> dict:
    with open(BASE_CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Search space definition
# ---------------------------------------------------------------------------
def suggest_hyperparams(trial: optuna.Trial) -> dict:
    """
    Define the hyperparameter search space.
    Modify ranges here to expand / narrow the search.
    """
    return {
        # Optimizer
        "learning_rate":            trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "backbone_lr_multiplier":   trial.suggest_float("backbone_lr_multiplier", 0.05, 0.3, log=True),
        "weight_decay":             trial.suggest_float("weight_decay", 0.0, 0.1),

        # Scheduler
        "warmup_ratio":             trial.suggest_float("warmup_ratio", 0.05, 0.2),

        # Batch
        "batch_size":               trial.suggest_categorical("batch_size", [16, 32]),
        "gradient_accumulation":    trial.suggest_categorical("gradient_accumulation", [1, 2, 4]),

        # NER head
        "ner_lstm_hidden":          trial.suggest_categorical("ner_lstm_hidden", [128, 256, 512]),
        "ner_dropout":              trial.suggest_float("ner_dropout", 0.1, 0.4),

        # POS head
        "pos_hidden_size":          trial.suggest_categorical("pos_hidden_size", [128, 256]),
        "pos_dropout":              trial.suggest_float("pos_dropout", 0.1, 0.4),

        # Coref head
        "coref_dropout":            trial.suggest_float("coref_dropout", 0.2, 0.5),

        # MTL loss
        "loss_init_sigma":          trial.suggest_float("loss_init_sigma", 0.5, 2.0),
    }


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------
def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective — returns the validation metric to MAXIMIZE.
    Currently: NER entity F1 on dev set.

    Replace the stub below with your actual training call once
    the MTL trainer is ready (Phase 3, Both students).
    """
    import wandb

    hp = suggest_hyperparams(trial)
    cfg = load_base_config()

    # Patch config with trial hyperparams
    cfg["optimizer"]["learning_rate"]           = hp["learning_rate"]
    cfg["optimizer"]["backbone_lr_multiplier"]  = hp["backbone_lr_multiplier"]
    cfg["optimizer"]["weight_decay"]            = hp["weight_decay"]
    cfg["scheduler"]["warmup_ratio"]            = hp["warmup_ratio"]
    cfg["training"]["batch_size"]               = hp["batch_size"]
    cfg["training"]["gradient_accumulation"]    = hp["gradient_accumulation"]
    cfg["ner"]["lstm_hidden"]                   = hp["ner_lstm_hidden"]
    cfg["ner"]["dropout"]                       = hp["ner_dropout"]
    cfg["pos"]["hidden_size"]                   = hp["pos_hidden_size"]
    cfg["pos"]["dropout"]                       = hp["pos_dropout"]
    cfg["coref"]["dropout"]                     = hp["coref_dropout"]
    cfg["loss"]["init_sigma"]                   = hp["loss_init_sigma"]

    run = wandb.init(
        project=cfg["wandb"]["project"],
        config={**hp, "trial_number": trial.number},
        tags=["hp_search"],
        reinit=True,
    )

    try:
        # ── STUB: replace with actual training call ──────────────────────
        # from src.training.trainer import Trainer
        # trainer = Trainer(cfg)
        # metrics = trainer.train()
        # val_ner_f1 = metrics["eval/ner_f1"]
        # ────────────────────────────────────────────────────────────────

        # Placeholder — remove when trainer is ready
        import random
        val_ner_f1 = random.uniform(0.70, 0.92)

        wandb.log({"eval/ner_f1": val_ner_f1, "trial": trial.number})
        run.finish()
        return val_ner_f1

    except Exception as e:
        run.finish()
        raise optuna.exceptions.TrialPruned() from e


# ---------------------------------------------------------------------------
# Pruner — stops unpromising trials early (saves GPU hours)
# ---------------------------------------------------------------------------
def get_pruner() -> optuna.pruners.BasePruner:
    return optuna.pruners.MedianPruner(
        n_startup_trials=5,      # don't prune the first 5 trials
        n_warmup_steps=3,        # don't prune within the first 3 epochs
        interval_steps=1,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna HP search for Arabic NLP MTL")
    parser.add_argument("--n_trials",   type=int,  default=30,             help="Number of Optuna trials")
    parser.add_argument("--study_name", type=str,  default="arabic_nlp_hp", help="Optuna study name")
    parser.add_argument("--storage",    type=str,  default=None,           help="Optuna DB URI (e.g. sqlite:///hp.db)")
    parser.add_argument("--direction",  type=str,  default="maximize",     help="maximize or minimize")
    args = parser.parse_args()

    study = optuna.create_study(
        study_name=args.study_name,
        direction=args.direction,
        pruner=get_pruner(),
        storage=args.storage,
        load_if_exists=True,   # resume interrupted search
    )

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # ── Results summary ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print("="*60)
    print(f"Best trial:  #{study.best_trial.number}")
    print(f"Best value:  {study.best_value:.4f}  (NER entity F1)")
    print("\nBest hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k:35s} = {v}")

    # Save best params to YAML for reproducibility
    best_cfg = load_base_config()
    best_cfg["_optuna_best_trial"] = study.best_trial.number
    best_cfg["_optuna_best_value"] = study.best_value
    out_path = Path("experiments/configs/best_hp.yaml")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(best_cfg, f, allow_unicode=True, default_flow_style=False)
    print(f"\n✓ Best config saved to {out_path}")


if __name__ == "__main__":
    main()