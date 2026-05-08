import mne
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from mne.decoding import Vectorizer

DATA_ROOT   = Path.home() / "brain_project" / "data" / "pre"
SUBJECTS    = [1, 2, 3, 5]
SOCIAL_IDS  = [111, 112, 113, 114]
NUMERIC_IDS = [125, 126, 127, 128]
WORD_IDS    = SOCIAL_IDS + NUMERIC_IDS


def load_subject(sub_id, tmin=0.0, tmax=0.5):
    sub  = f"{sub_id:02d}"
    path = DATA_ROOT / sub / \
        f"sub-{sub}_ses-eeg_task-innerspeech_desc-words_epo.fif"
    epochs = mne.read_epochs(path, preload=True, verbose=False)
    epochs = epochs.crop(tmin=tmin, tmax=tmax)
    # Get EEG data and labels
    X      = epochs.get_data(picks="eeg").astype(np.float64)
    # y_raw is the original event ID (e.g. 111 for "child", 125 for "four")
    y_raw  = epochs.events[:, 2]
    # Filter to only keep trials corresponding to our 8 words
    keep   = np.isin(y_raw, WORD_IDS)
    X      = X[keep]
    y_raw  = y_raw[keep]
    y      = np.where(np.isin(y_raw, SOCIAL_IDS), 0, 1)
    return X, y

# Cross-validation
# StratifiedKFold is used to maintain class balance in each fold.
cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rng = np.random.default_rng(42)

pipe = Pipeline([
    ("vec",    Vectorizer()),
    ("scaler", StandardScaler()),
    ("svm",    SVC(kernel="linear", C=0.001))
])

# Number of permutations for the test
N_PERMS = 100

print("Permutation test — Linear SVM C=0.001, window 0-500ms")
print(f"Running {N_PERMS} permutations per subject...\n")

for sub_id in SUBJECTS:
    X, y = load_subject(sub_id)

    # True accuracy
    true_acc = cross_val_score(pipe, X, y, cv=cv).mean() * 100

    # Permuted accuracies
    perm_accs = []
    for i in range(N_PERMS):
        y_shuf = rng.permutation(y)
        acc    = cross_val_score(pipe, X, y_shuf, cv=cv).mean() * 100
        perm_accs.append(acc)
        if (i + 1) % 50 == 0:
            print(f"  sub-{sub_id:02d}: {i+1}/{N_PERMS} perms done...",
                  flush=True)

    perm_accs = np.array(perm_accs)

    print(f"\n  sub-{sub_id:02d} results:")
    print(f"    True accuracy   : {true_acc:.2f}%")
    print(f"    Permuted mean   : {perm_accs.mean():.2f}%")

print("Chance level: 50.0%")