import mne
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold
from mne.decoding import CSP
from pathlib import Path


# 1. Load data
root_folder = Path(__file__).resolve().parent.parent
file_path = root_folder / "EEG-proc" / "sub-03_cleaned-epo.fif"
epochs = mne.read_epochs(file_path, preload=True)

# 2. Prepare labels (y)
# We map: Social words -> 0, Numeric words -> 1
# We ignore fixation (1) and rest (2) for this analysis
event_id_map = {
    'Social/Child': 111, 'Social/Daughter': 112, 'Social/Father': 113, 'Social/Wife': 114,
    'Numeric/Ten': 125, 'Numeric/Three': 126, 'Numeric/Six': 127, 'Numeric/Four': 128
}
epochs.event_id = event_id_map

# Select only the word epochs
epochs_data = epochs['Social', 'Numeric']
X = epochs_data.get_data(picks='eeg') # Format: [trials, channels, times]
y = epochs_data.events[:, 2]

# Convert IDs to 0 and 1 for easier analysis
y = np.where(np.isin(y, [111, 112, 113, 114]), 0, 1)

print(f"Data ready for AI: {X.shape[0]} trials, {X.shape[1]} channels.")

# 3. Build the ML pipeline
# CSP extracts spatial patterns, LDA classifies them
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
lda = LinearDiscriminantAnalysis()

clf = Pipeline([('CSP', csp), ('LDA', lda)])

# 4. Evaluate with Cross-Validation (K-fold)
# We split the data into 5 parts, train on 4 and test on 1, repeat 5 times.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y, cv=cv, n_jobs=None)

# 5. Results
print("\n--- Results ---")
print(f"Accuracy (mean): {np.mean(scores) * 100:.2f}%")
print(f"Chance level: 50.00%")
print(f"Standard deviation: {np.std(scores) * 100:.2f}%")