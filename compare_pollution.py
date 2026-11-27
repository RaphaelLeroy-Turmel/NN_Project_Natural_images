"""Show training images before and after pollution, in one grid.

    python show_pollution.py

Top row : original images. Bottom row : the same images after pollution.
Only images that are really polluted are shown (chosen with the same seed as the runs).
The figure is also saved in results/figures/.
"""
import matplotlib.pyplot as plt
import numpy as np

from src.data_load import load_images
from src.degradation import pollute_train
from src.runs import RESULTS_DIR

# ---- Settings -----------------------------------------------------------------
POLLUTION_METHOD = "masque"   # "flou", "pixelisation" or "masque"
POLLUTION_FRACTION = 0.6      # fraction of the TRAIN images that are degraded (0.3 = 30 %)
SEED = 44                     # same seed as a run -> same polluted images as in that run
N_IMAGES = 10                 # number of images shown
# -------------------------------------------------------------------------------


def main():
    X, y, classes = load_images("train")
    X_polluted = pollute_train(X, POLLUTION_METHOD, POLLUTION_FRACTION, SEED)

    # Images that really changed (pollute_train only modifies a fraction of them)
    changed = np.flatnonzero(np.any(X_polluted != X, axis=(1, 2, 3)))
    print(f"Method {POLLUTION_METHOD}, {POLLUTION_FRACTION:.0%} asked : "
          f"{len(changed)} / {len(X)} training images modified")
    if len(changed) == 0:
        raise SystemExit("No image was modified : check POLLUTION_FRACTION.")

    rng = np.random.default_rng(SEED)
    shown = rng.choice(changed, size=min(N_IMAGES, len(changed)), replace=False)

    fig, axes = plt.subplots(2, len(shown), figsize=(1.8 * len(shown), 4.2), squeeze=False)
    for col, i in enumerate(shown):
        for row, images in enumerate((X, X_polluted)):
            ax = axes[row][col]
            ax.imshow(images[i])
            ax.set_xticks([])
            ax.set_yticks([])
        axes[0][col].set_title(classes[y[i]], fontsize=9)
    axes[0][0].set_ylabel("Before", fontsize=10)
    axes[1][0].set_ylabel("After", fontsize=10)
    fig.suptitle(f"Training images before / after pollution : {POLLUTION_METHOD}, "
                 f"{POLLUTION_FRACTION:.0%} of the train set, seed {SEED}", fontsize=11)
    fig.tight_layout()

    out_dir = RESULTS_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"pollution_examples_{POLLUTION_METHOD}{round(POLLUTION_FRACTION * 100)}_seed{SEED}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("Saved :", out)
    plt.show()


if __name__ == "__main__":
    main()