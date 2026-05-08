import mne
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold
from mne.decoding import CSP


# 1. Load preprocessed word epochs
file_path = Path.home() / "brain_project" / "data" / "pre" / "03" / \
    "sub-03_ses-eeg_task-innerspeech_desc-words_epo.fif"

epochs = mne.read_epochs(file_path, preload=True)
epochs = epochs.crop(tmin=0.0, tmax=2.0)

print(epochs)
print("Event IDs:", epochs.event_id)


# 2. Keep EEG data only
X = epochs.get_data(picks="eeg")   # shape: trials × channels × time
y_raw = epochs.events[:, 2]


# 3. Convert labels to binary
social_ids = [111, 112, 113, 114]
numeric_ids = [125, 126, 127, 128]

keep = np.isin(y_raw, social_ids + numeric_ids)

X = X[keep]
y_raw = y_raw[keep]

y = np.where(np.isin(y_raw, social_ids), 0, 1)

print(f"Trials: {X.shape[0]}")
print(f"Channels: {X.shape[1]}")
print(f"Time points: {X.shape[2]}")
print(f"Labels: Social={np.sum(y == 0)}, Numeric={np.sum(y == 1)}")


# 4. CSP + LDA model
clf = Pipeline([
    ("CSP", CSP(n_components=4, reg=0.1, log=True, norm_trace=False)),
    ("LDA", LinearDiscriminantAnalysis())
])


# 5. Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(clf, X, y, cv=cv)


# 6. Results
print("\n--- Results ---")
print(f"Accuracy mean: {scores.mean() * 100:.2f}%")
print(f"Accuracy std:  {scores.std() * 100:.2f}%")
print("Chance level: 50.00%")