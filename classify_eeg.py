import mne
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold
from mne.decoding import CSP

# 1. Ladda data
file_path = "C:/projects/brain_imaging/EEG-proc/sub-01_cleaned-epo.fif"
epochs = mne.read_epochs(file_path, preload=True)

# 2. Förbered etiketter (y)
# Vi mappar: Sociala ord -> 0, Numeriska ord -> 1
# Vi struntar i fixation (1) och vila (2) för denna analys
event_id_map = {
    'Social/Child': 111, 'Social/Daughter': 112, 'Social/Father': 113, 'Social/Wife': 114,
    'Numeric/Ten': 125, 'Numeric/Three': 126, 'Numeric/Six': 127, 'Numeric/Four': 128
}
epochs.event_id = event_id_map

# Välj ut endast ord-epokerna
epochs_data = epochs['Social', 'Numeric']
X = epochs_data.get_data(picks='eeg') # Format: [trials, channels, times]
y = epochs_data.events[:, 2]

# Gör om ID:n till 0 och 1 för enklare analys
y = np.where(np.isin(y, [111, 112, 113, 114]), 0, 1)

print(f"Data redo för AI: {X.shape[0]} försök, {X.shape[1]} kanaler.")

# 3. Bygg ML-pipelinen
# CSP extraherar spatiala mönster, LDA klassificerar dem
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
lda = LinearDiscriminantAnalysis()

clf = Pipeline([('CSP', csp), ('LDA', lda)])

# 4. Utvärdera med Cross-Validation (K-fold)
# Vi delar upp datan i 5 delar, tränar på 4 och testar på 1, upprepar 5 gånger.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y, cv=cv, n_jobs=None)

# 5. Resultat
print("\n--- Resultat ---")
print(f"Accuracy (medel): {np.mean(scores) * 100:.2f}%")
print(f"Slumpens chans: 50.00%")
print(f"Standardavvikelse: {np.std(scores) * 100:.2f}%")