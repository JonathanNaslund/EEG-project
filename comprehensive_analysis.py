import mne
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

def run_comprehensive_analysis(selected_epochs, title_suffix=""):
    # 1. Map Events
    new_event_id = {
        'Social/Child': 111, 'Social/Daughter': 112, 'Social/Father': 113, 'Social/Wife': 114,
        'Numeric/Ten': 125, 'Numeric/Three': 126, 'Numeric/Six': 127, 'Numeric/Four': 128
    }
    actual_events = np.unique(selected_epochs.events[:, 2])
    selected_epochs.event_id = {k: v for k, v in new_event_id.items() if v in actual_events}

    # 2. Create Averages
    evoked_social = selected_epochs['Social'].average()
    evoked_numeric = selected_epochs['Numeric'].average()
    evoked_diff = mne.combine_evoked([evoked_social, evoked_numeric], weights=[1, -1])

    # 3. NUMERICAL OUTPUT (The data for me to analyze)
    print(f"\n" + "="*60)
    print(f" STATISTICAL SUMMARY: {title_suffix}")
    print("="*60)
    
    # We define ROIs based on your specific observations (NW, Left, NE)
    rois = {
        'NW_Frontal (Social?)': ['Fp1', 'F3', 'F7', 'AF3'],
        'Left_Temporal (Speech?)': ['T7', 'P7', 'TP7'],
        'NE_Frontal': ['Fp2', 'F4', 'F8', 'AF4'],
        'Central': ['Cz', 'CPz', 'FCz']
    }

    times_to_measure = [0.3, 0.4, 0.5] # Seconds
    
    print(f"{'Region':25} | {'Time':5} | {'Social':8} | {'Numeric':8} | {'Diff'}")
    print("-" * 65)

    for t in times_to_measure:
        # We take a small window around the time point (e.g., 50ms)
        t_start, t_end = t - 0.025, t + 0.025
        
        for roi_name, channels in rois.items():
            # Get only channels that exist in this subject
            chans = [c for c in channels if c in evoked_social.ch_names]
            if chans:
                # Calculate mean microvolts
                s_val = evoked_social.copy().crop(t_start, t_end).pick(chans).data.mean() * 1e6
                n_val = evoked_numeric.copy().crop(t_start, t_end).pick(chans).data.mean() * 1e6
                d_val = s_val - n_val
                print(f"{roi_name:25} | {t:4.1f}s | {s_val:6.2f} µV | {n_val:6.2f} µV | {d_val:6.2f} µV")

    # 4. VISUALIZATION
    # Clean EXG for plotting
    for ev in [evoked_social, evoked_numeric, evoked_diff]:
        ev.pick_types(eeg=True, exclude=['EXG7', 'EXG8'])

    times_plot = [0.3, 0.4, 0.5, 0.6]
    
    # Show the Difference map as the primary focus
    fig = evoked_diff.plot_topomap(times=times_plot, colorbar=True, units='µV', res=128)
    fig.suptitle(f"Spatial Contrast (Social - Numeric) {title_suffix}")
    
    plt.show()

if __name__ == "__main__":
    root_folder = Path(__file__).resolve().parent
    proc_dir = root_folder / "EEG-proc"
    files = sorted(list(proc_dir.glob("sub-*_cleaned-epo.fif")))

    print("\n--- DATA ANALYZER ---")
    for i, f in enumerate(files):
        print(f"[{i+1}] {f.name}")
    print(f"[{len(files)+1}] ALL SUBJECTS")
    
    choice = input("\nSelect Subject ID or 'all': ").strip().lower()

    if choice == 'all' or choice == str(len(files)+1):
        all_data = [mne.read_epochs(f, preload=True, verbose=False) for f in files]
        combined = mne.concatenate_epochs(all_data)
        run_comprehensive_analysis(combined, title_suffix="GRAND_AVERAGE")
    else:
        idx = int(choice) - 1
        epochs = mne.read_epochs(files[idx], preload=True, verbose=False)
        run_comprehensive_analysis(epochs, title_suffix=files[idx].name[:6])