from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

CLASSES = [
    "SAFE",
    "WEATHER",
    "ALCOHOL",
    "EXPLOSIVE",
    "NARCOTIC",
]

# Current 12,000-sample internal validation confusion matrices.
# Rows = true class, columns = predicted class.
RF_CM = np.array([
    [1978, 136,   5, 108, 173],
    [   7, 2187,   2,   1, 203],
    [   0,   19, 2218,   6, 157],
    [ 143,   34,   5, 1936, 282],
    [  38,  422,  61,  96, 1783],
])

TINYML_CM = np.array([
    [1994, 104,  11, 144, 147],
    [  22, 2143,   6,   4, 225],
    [   1,   14, 2262,   6, 117],
    [  85,   28,   7, 2022, 258],
    [  44,  341,  63, 120, 1832],
])


def plot_cm(ax, cm, title):
    cmap = plt.get_cmap("Blues")
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            cell_color = cmap(cm[i, j] / max(1, cm.max()))
            red, green, blue, _ = cell_color
            luminance = 0.299 * red + 0.587 * green + 0.114 * blue
            text_color = "white" if luminance < 0.55 else "black"

            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
                weight="bold",
            )

    return im


def main():
    if RF_CM.sum() != 12000:
        raise ValueError("Random Forest confusion matrix must contain 12,000 samples.")

    if TINYML_CM.sum() != 12000:
        raise ValueError("TinyML confusion matrix must contain 12,000 samples.")

    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))

    plot_cm(axes[0], RF_CM, "Random Forest — Validation Confusion Matrix")
    plot_cm(axes[1], TINYML_CM, "TinyML — Validation Confusion Matrix")

    fig.suptitle(
        "SENTRY Model Comparison — Confusion Matrices",
        fontsize=16,
        fontweight="bold",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = output_dir / "confusion_matrices.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved confusion matrix plot to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
