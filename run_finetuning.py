"""Track C launcher : load the data and run the fine-tuning experiments.

Run from the root of the repository : python run_finetuning.py
"""
from src.data_load import load_images
from src.experiment import run_experiment

if __name__ == "__main__":
    # Raw uint8 pixels : normalization is done inside run_experiment
    X_train, y_train, class_names = load_images("train")
    X_valid, y_valid, _ = load_images("validation")  # check the split name used in main.py
    X_test, y_test, _ = load_images("test")
    data = (X_train, y_train, X_valid, y_valid, X_test, y_test, class_names)

    # Quick check first : few epochs, to verify that results/ is filled correctly
    # run_experiment(*data, epochs_head=2)

    run_experiment(*data)                                # frozen backbone (baseline)
    run_experiment(*data, unfreeze_last_n_layers=30)     # fine-tuning

    # Pollution study : one run per level
    for fraction in (0.1, 0.2, 0.3):
        run_experiment(*data, unfreeze_last_n_layers=30,
                       pollution_method="flou", pollution_fraction=fraction)    