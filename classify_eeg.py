import os
import mne
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold
from mne.decoding import Vectorizer, CSP
from sklearn.preprocessing import StandardScaler

def run_classification(epochs, title_suffix=""):
    """Runs the CSP + LDA pipeline on the provided epochs."""
    
    # 1. Map Events
    event_id_map = {
        'Social/Child': 111, 'Social/Daughter': 112, 'Social/Father': 113, 'Social/Wife': 114,
        'Numeric/Ten': 125, 'Numeric/Three': 126, 'Numeric/Six': 127, 'Numeric/Four': 128
    }
    # Filter map to only include events present in this specific data
    actual_events = np.unique(epochs.events[:, 2])
    epochs.event_id = {k: v for k, v in event_id_map.items() if v in actual_events}

    # 2. Time Cropping (The Hypothesis-Driven Step)
    # We focus on the semantic peak (300ms - 500ms), change to any tmin/tmax values between -0.2 to 0.8
    print(f"\n--- Decoding Task: {title_suffix} ---")
    print("Focusing on Semantic Window (0.3s - 0.5s)...")
    
    # Selecting Social/Numeric and cropping
    word_epochs = epochs['Social', 'Numeric'].copy().crop(tmin=0.3, tmax=0.5)
    
    # Using all channels
    X = word_epochs.get_data(picks='eeg')  # [trials, channels, times]
    y = word_epochs.events[:, 2]

    # Pruning: Focus only on Frontal and Temporal sensors
    # This removes Occipital (visual) noise
    #word_epochs.pick([
    #    ch for ch in word_epochs.ch_names if ch.startswith(('F', 'T'))
    #])

    # Get Data and Labels
    #X = word_epochs.get_data()  
    #y = word_epochs.events[:, 2]
    
    y = np.where(np.isin(y, [111, 112, 113, 114]), 0, 1)

    # 3. Build the ML Pipeline (using Vectorizer instead)
    # CSP finds the 'spatial fingerprint' of Social vs Numeric
    #csp = CSP(n_components=4, reg='ledoit_wolf', log=True, norm_trace=False, rank='info')
    #lda = LinearDiscriminantAnalysis()
    #clf = Pipeline([('CSP', csp), ('LDA', lda)])

    # Replace CSP with Vectorizer
    clf = Pipeline([
    ('Vectorizer', Vectorizer()), 
    ('Scaler', StandardScaler()), # Normalizes the µV values
    ('LDA', LinearDiscriminantAnalysis())
    ])

    # 4. Evaluate with Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, n_jobs=None)

    # 5. Output Results
    print(f"Trials: {len(y)} ({np.sum(y==0)} Social, {np.sum(y==1)} Numeric)")
    print(f"Accuracy: {np.mean(scores) * 100:.2f}% (+/- {np.std(scores) * 100:.2f}%)")
    return scores

if __name__ == "__main__":
    root_folder = Path(__file__).resolve().parent
    proc_dir = root_folder / "EEG-proc"
    files = sorted(list(proc_dir.glob("sub-*_cleaned-epo.fif")))

    if not files:
        print(f"No processed files found in {proc_dir}!")
        exit()

    print("\n--- EEG CLASSIFICATION MENU ---")
    for i, f in enumerate(files):
        print(f"[{i+1}] {f.name}")
    print(f"[{len(files)+1}] ALL (Combine all subjects into one giant model)")
    print(f"[{len(files)+2}] INDIVIDUALS (Run each subject separately one by one)")
    
    choice = input("\nSelect option: ").strip().lower()

    try:
        # Option: GRAND AVERAGE MODEL
        if choice == str(len(files)+1):
            print("Loading all subjects for a combined model...")
            all_data = [mne.read_epochs(f, preload=True, verbose=False) for f in files]
            combined_epochs = mne.concatenate_epochs(all_data)
            run_classification(combined_epochs, title_suffix="Grand Average")

        # Option: RUN EACH INDIVIDUALLY
        elif choice == str(len(files)+2) or choice == 'all':
            global_scores = []
            for f in files:
                epochs = mne.read_epochs(f, preload=True, verbose=False)
                res = run_classification(epochs, title_suffix=f.name[:6])
                global_scores.append(np.mean(res))
            print(f"\n{'='*30}")
            print(f"OVERALL AVG ACCURACY: {np.mean(global_scores)*100:.2f}%")
            print(f"{'='*30}")

        # Option: SINGLE SUBJECT
        else:
            idx = int(choice) - 1
            epochs = mne.read_epochs(files[idx], preload=True, verbose=False)
            run_classification(epochs, title_suffix=files[idx].name[:6])

    except Exception as e:
        print(f"An error occurred: {e}")