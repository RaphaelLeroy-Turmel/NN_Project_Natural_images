"""Launch experiments, each one in its own process.

Why : a crash only loses the run in progress, and the memory (TensorFlow, arrays) is
released at the end of every run. Finished runs are skipped, so after a crash it is enough
to launch the same command again.

This module does not import TensorFlow : the heavy imports happen in the child processes.
"""
import multiprocessing as mp

from src.runs import has_test, is_done, run_name

VALID_SPLIT = "validation"  # name of the validation split in data_load.load_images


def _train_worker(config):
    # Heavy imports here, inside the child process only
    from src.data_load import load_images
    from src.experiment import run_experiment

    X_train, y_train, class_names = load_images("train")
    X_valid, y_valid, _ = load_images(VALID_SPLIT)
    extra = {}
    if config.get("evaluate_test"):  # the test set is loaded only when it is needed
        X_test, y_test, _ = load_images("test")
        extra = {"X_test": X_test, "y_test": y_test}
    run_experiment(X_train, y_train, X_valid, y_valid, class_names, **extra, **config)


def run_all(configs):
    """Run every config (a list of dicts of run_experiment parameters). Returns the failed names."""
    context = mp.get_context("spawn")  # the default on Windows, explicit for every system
    failed = []
    for i, config in enumerate(configs, start=1):
        name = run_name(**config)
        if is_done(config) and (not config.get("evaluate_test") or has_test(config)):
            print(f"[{i}/{len(configs)}] skip {name} (already done)")
            continue
        print(f"[{i}/{len(configs)}] run  {name}")
        process = context.Process(target=_train_worker, args=(config,))
        process.start()
        process.join()
        if process.exitcode != 0:
            print(f"   FAILED (exit code {process.exitcode}) : {name}")
            failed.append(name)

    print(f"\n{len(configs) - len(failed)}/{len(configs)} runs OK")
    if failed:
        print("Launch the same command again to retry (finished runs are skipped). Failed :")
        for name in failed:
            print("  ", name)
    return failed
