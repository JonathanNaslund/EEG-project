import mne
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from mne.decoding import Vectorizer
import matplotlib.pyplot as plt

# Configuration
DATA_ROOT   = Path.home() / "brain_project" / "data" / "pre"
SUBJECTS    = [1, 2, 3, 5]
SOCIAL_IDS  = [111, 112, 113, 114]
NUMERIC_IDS = [125, 126, 127, 128]
WORD_IDS    = SOCIAL_IDS + NUMERIC_IDS
WORD_NAMES  = {
    111: "child", 112: "daughter", 113: "father", 114: "wife",
    125: "four",  126: "three",    127: "ten",     128: "six"
}

# Optimised parameters (found through extensive search)
TMIN       = 0.0    # start of time window (seconds)
TMAX       = 0.5    # end of time window (seconds)
SVM_C      = 0.001  # regularisation — low C = strong regularisation
SVM_KERNEL = "linear"


# Data loading
def load_subject(sub_id, task):
    """
    Load one subject's EEG data cropped to the optimal time window.
    task = 'binary' → y is 0 (social) or 1 (numeric)
    task = 'word'   → y is the original word ID (111-128)
    """
    sub    = f"{sub_id:02d}"
    path   = DATA_ROOT / sub / \
        f"sub-{sub}_ses-eeg_task-innerspeech_desc-words_epo.fif"
    epochs = mne.read_epochs(path, preload=True, verbose=False)
    epochs = epochs.crop(tmin=TMIN, tmax=TMAX)
    # Get EEG data and labels
    X     = epochs.get_data(picks="eeg").astype(np.float64)
    # y_raw is the original event ID (e.g. 111 for "child", 125 for "four")
    y_raw = epochs.events[:, 2]
    # Filter to only keep trials corresponding to our 8 words
    keep  = np.isin(y_raw, WORD_IDS)
    X     = X[keep]
    y_raw = y_raw[keep]

    if task == "binary":
        y = np.where(np.isin(y_raw, SOCIAL_IDS), 0, 1)
    else:
        y = y_raw.copy()

    return X, y


# Model
def make_model():
    """
    Linear SVM pipeline:
      1. Vectorizer  — flattens (channels x time) into a 1D feature vector
      2. StandardScaler — normalises each feature to zero mean, unit variance
      3. Linear SVM  — finds the maximum margin hyperplane between classes
    """
    return Pipeline([
        ("vectorizer", Vectorizer()),
        ("scaler",     StandardScaler()),
        ("svm",        SVC(kernel=SVM_KERNEL, C=SVM_C))
    ])


# Cross-validation
# StratifiedKFold is used to maintain class balance in each fold.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# Run both tasks
tasks = {
    "binary": {
        "label":   "2-class: Social vs Numeric",
        "chance":  50.0,
        "labels":  ["Social", "Numeric"],
        "ids":     [0, 1],
    },
    "word": {
        "label":   "8-class: Word identity",
        "chance":  12.5,
        "labels":  [WORD_NAMES[w] for w in WORD_IDS],
        "ids":     WORD_IDS,
    },
}

# Set up the plot for confusion matrices
fig, ax = plt.subplots(1, 1, figsize=(9, 7))


for task, info in tasks.items():

    n_cls      = len(info["ids"])
    cm_total   = np.zeros((n_cls, n_cls), dtype=int)
    sub_accs   = []

    print(f"\n{info['label']}")

    for sub_id in SUBJECTS:
        X, y   = load_subject(sub_id, task)
        model  = make_model()
        y_pred = cross_val_predict(model, X, y, cv=cv)
        acc    = np.mean(y_pred == y) * 100
        sub_accs.append(acc)
        cm_total += confusion_matrix(y, y_pred, labels=info["ids"])
        print(f"  sub-{sub_id:02d}: {acc:.2f}%")

    mean_acc = np.mean(sub_accs)
    print(f"  Mean  : {mean_acc:.2f}%  (chance {info['chance']:.1f}%)")

    # Plot confusion matrix
    if task == "word":
        # Normalise confusion matrix to percentages
        cm_norm = cm_total.astype(float) / cm_total.sum(axis=1, keepdims=True)
        disp    = ConfusionMatrixDisplay(
            confusion_matrix=cm_norm,
            display_labels=info["labels"]
        )

        # Add colorbar
        disp.plot(
            ax=ax,
            colorbar=(task == "word"),
            cmap="Blues",
            values_format=".2f",
            xticks_rotation=45
        )

        # Red dashed line separating social and numeric blocks
        if task == "word":
            ax.axhline(3.5, color="red", linewidth=2, linestyle="--")
            ax.axvline(3.5, color="red", linewidth=2, linestyle="--")

        # Annotate with subject accuracies
        ax.set_title(
            f"{info['label']}\n"
            f"Mean: {mean_acc:.2f}%  |  Chance: {info['chance']:.1f}%  |  "
            f"Gap: {mean_acc - info['chance']:+.2f}pp",
            fontsize=11, fontweight="bold"
        )
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)

plt.tight_layout()
plt.savefig("final_model_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved: final_model_results.png")