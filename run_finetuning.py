"""Track C launcher : the steps of the common experimental protocol.

Run from the root of the repository, one step at a time :

    python run_finetuning.py tune        # step 1 : tuning on the validation set (test NOT read)
    python run_finetuning.py summary     # validation results, mean +- std over the seeds
    python run_finetuning.py seeds       # step 3 : shortlisted variants with the 3 seeds (test NOT read)
    python run_finetuning.py test        # step 4 : finalists measured on the test set (once)
    python run_finetuning.py pollution   # step 5 : finalists replayed on polluted training data

If the process crashes, launch the same command again : finished runs are skipped.
"""
import sys

from src.runner import run_all
from src.runs import summarize_runs

SEEDS = (42, 43, 44)  # same seeds for every track (to be validated by the team)

# Pollution study (common to the three tracks, to be validated by the team)
POLLUTION_METHOD = "masque"
POLLUTION_LEVELS = ( 0.3, 0.5, 0.7, 0.8)  # fraction of the TRAIN images that are degraded

# Step 1 : variants to tune. One run per config, with the first seed.
# Number of unfrozen layers : 0 = frozen backbone (about 10 layers = 1 block).
TUNING_GRID = (# 6 models in total
    [dict(unfreeze_last_n_layers=0, lr_head=lr) for lr in (1e-3, 1e-4)]
    + [dict(unfreeze_last_n_layers=n, lr_finetune=lr) for n in (10, 30, 60) for lr in (1e-4, 1e-5)]
)


# Step 3 : the best variants of each category after step 1 (one or two per category).
# Fill this list by hand after `summary`, for example :
#     dict(unfreeze_last_n_layers=30, lr_finetune=1e-5),
SHORTLIST = [
    dict(unfreeze_last_n_layers=0, lr_head=1e-3),        # gelé
    dict(unfreeze_last_n_layers=30, lr_finetune=1e-4),   # fine-tuning : meilleure loss
    dict(unfreeze_last_n_layers=10, lr_finetune=1e-4),   # fine-tuning : plus simple, le moins d'erreurs
]

# Step 4 : ONE finalist per category, chosen with the validation set (loss, then F1 macro,
# then the simplicity rule). Share this list with the team BEFORE running the `test` step.
FINALISTS = [
    dict(unfreeze_last_n_layers=0, lr_head=1e-3),        # backbone gelé
    dict(unfreeze_last_n_layers=30, lr_finetune=1e-4),   # fine-tuning
]


def require(configs, name):
    if not configs:
        sys.exit(f"{name} is empty : fill it in run_finetuning.py first (see the protocol).")


def main(step):
    if step == "tune":
        run_all([{**config, "seed": SEEDS[0]} for config in TUNING_GRID])

    elif step == "summary":
        import pandas as pd

        pd.set_option("display.width", 200)
        pd.set_option("display.max_columns", 20)
        print(summarize_runs())

    elif step == "seeds":
        require(SHORTLIST, "SHORTLIST")
        run_all([{**config, "seed": seed} for config in SHORTLIST for seed in SEEDS])

    elif step == "test":
        require(FINALISTS, "FINALISTS")
        print("The test set is read here, once per finalist and seed.")
        run_all([{**config, "seed": seed, "evaluate_test": True}
                 for config in FINALISTS for seed in SEEDS])

    elif step == "pollution":
        require(FINALISTS, "FINALISTS")
        # The 0 % level is the clean finalist already measured by the `test` step.
        # The pollution seed is the run seed : same polluted images for every category.
        run_all([
            {**config, "seed": seed, "pollution_method": POLLUTION_METHOD,
             "pollution_fraction": level, "evaluate_test": True}
            for config in FINALISTS for level in POLLUTION_LEVELS for seed in SEEDS
        ])

    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
