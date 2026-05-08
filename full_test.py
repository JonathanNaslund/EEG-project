import mne
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from mne.decoding import Vectorizer
import itertools


# Configuration
DATA_ROOT = Path.home() / "brain_project" / "data" / "pre"

SUBJECTS = [1, 2, 3, 5]

SOCIAL_IDS  = [111, 112, 113, 114]
NUMERIC_IDS = [125, 126, 127, 128]
WORD_IDS    = SOCIAL_IDS + NUMERIC_IDS

TMIN = 0.0
TMAX = 0.5

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# Data loading
def load_subject(sub_id, tmin=TMIN, tmax=TMAX, task="binary"):
    sub = f"{sub_id:02d}"
    path = DATA_ROOT / sub / f"sub-{sub}_ses-eeg_task-innerspeech_desc-words_epo.fif"
    epochs = mne.read_epochs(path, preload=True, verbose=False)
    epochs = epochs.crop(tmin=tmin, tmax=tmax)
    # Get EEG data and labels
    X = epochs.get_data(picks="eeg").astype(np.float64)
    # y_raw is the original event ID (e.g. 111 for "child", 125 for "four")
    y_raw = epochs.events[:, 2]
    # Filter to only keep trials corresponding to our 8 words
    keep = np.isin(y_raw, WORD_IDS)
    X = X[keep]
    y_raw = y_raw[keep]
    if task == "binary":
        y = np.where(np.isin(y_raw, SOCIAL_IDS), 0, 1)
    else:
        y = y_raw.copy()

    return X, y


# Evaluation
def evaluate(task, svm):
    accs = []

    for sub_id in SUBJECTS:
        X, y = load_subject(sub_id, task=task)

        pipe = Pipeline([
            ("vec", Vectorizer()),
            ("scaler", StandardScaler()),
            ("svm", svm)
        ])

        acc = cross_val_score(pipe, X, y, cv=cv).mean() * 100
        accs.append(acc)

    return np.mean(accs), np.std(accs), accs


# Candidate SVM configurations
candidates = {}

# Linear
for C in [0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]:
    name = f"linear | C={C}"
    candidates[name] = SVC(kernel="linear", C=C)

# RBF
for C, gamma in itertools.product(
    [0.01, 0.1, 1, 10, 100],
    ["scale", "auto", 0.0001, 0.001, 0.01]
):
    name = f"rbf | C={C}, gamma={gamma}"
    candidates[name] = SVC(kernel="rbf", C=C, gamma=gamma)

# Polynomial
for C, gamma, degree in itertools.product(
    [0.0001, 0.001, 0.01, 0.1],
    ["scale", "auto"],
    [2, 3, 4]
):
    name = f"poly | C={C}, gamma={gamma}, degree={degree}"
    candidates[name] = SVC(kernel="poly", C=C, gamma=gamma, degree=degree)

# Sigmoid
for C, gamma in itertools.product(
    [0.0001, 0.001, 0.01, 0.1],
    ["scale", "auto"]
):
    name = f"sigmoid | C={C}, gamma={gamma}"
    candidates[name] = SVC(kernel="sigmoid", C=C, gamma=gamma)


print(f"Total configurations: {len(candidates)}")
print(f"Total model evaluations: {len(candidates) * 2 * len(SUBJECTS)}")


# Run search
tasks = [
    {
        "task": "binary",
        "label": "2-class Social vs Numeric",
        "prev_best": 63.31,
        "chance": 50.0,
    },
    {
        "task": "word",
        "label": "8-class Word identity",
        "prev_best": 22.33,
        "chance": 12.5,
    },
]

all_task_results = {}

for task_info in tasks:
    task = task_info["task"]
    label = task_info["label"]
    prev_best = task_info["prev_best"]
    chance = task_info["chance"]

    print(f"\nRunning search for: {label}")

    task_results = {}
    best_mean = -np.inf
    best_name = None

    for i, (name, svm) in enumerate(candidates.items(), 1):
        mean, std, accs = evaluate(task, svm)

        task_results[name] = {
            "mean": mean,
            "std": std,
            "accs": accs,
        }

        if mean > best_mean:
            best_mean = mean
            best_name = name

        # Small progress update
        if i % 10 == 0:
            print(f"  {i}/{len(candidates)} configs done")

    all_task_results[task] = {
        "label": label,
        "results": task_results,
        "best_name": best_name,
        "best_mean": best_mean,
        "prev_best": prev_best,
        "chance": chance,
    }

    print(f"  Best config : {best_name}")
    print(f"  Best mean   : {best_mean:.2f}%")
    print(f"  Previous    : {prev_best:.2f}%")
    print(f"  Improvement : {best_mean - prev_best:+.2f}pp")
    print(f"  Gap chance  : {best_mean - chance:+.2f}pp")


# Final summary
print("\nFINAL SUMMARY")

for task, info in all_task_results.items():
    print(f"\n{info['label']}")
    print(f"  Best config   : {info['best_name']}")
    print(f"  Best accuracy : {info['best_mean']:.2f}%")
    print(f"  Previous best : {info['prev_best']:.2f}%")
    print(f"  Improvement   : {info['best_mean'] - info['prev_best']:+.2f}pp")
    print(f"  Gap vs chance : {info['best_mean'] - info['chance']:+.2f}pp")
