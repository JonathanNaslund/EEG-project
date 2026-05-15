import mne
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import time

def run_visualization(selected_epochs, title_suffix="", vlim=(-10, 5)):
    """Kör själva visualiseringslogiken på de valda epokerna."""
    # 1. Map Events
    new_event_id = {
        'Social/Child': 111, 'Social/Daughter': 112, 'Social/Father': 113, 'Social/Wife': 114,
        'Numeric/Ten': 125, 'Numeric/Three': 126, 'Numeric/Six': 127, 'Numeric/Four': 128
    }
    
    actual_events = np.unique(selected_epochs.events[:, 2])
    selected_epochs.event_id = {k: v for k, v in new_event_id.items() if v in actual_events}

    print(f"--- Genererar visualisering för {title_suffix} ---")
    
    # 2. Skapa genomsnitt (Evoked)
    evoked_social = selected_epochs['Social'].average()
    evoked_numeric = selected_epochs['Numeric'].average()
    evoked_diff = mne.combine_evoked([evoked_social, evoked_numeric], weights=[1, -1])

    # --- NEW: DETAILED NUMERICAL OUTPUT SECTION ---
    print(f"\n{'='*75}")
    print(f" DATA SUMMARY TABLE: {title_suffix}")
    print(f"{'='*75}")
    print(f"{'Region':20} | {'Time':6} | {'Social':9} | {'Numeric':9} | {'Diff (S-N)'}")
    print(f"{'-'*75}")

    rois = {
        'Frontal Left (NW)': ['Fp1', 'F3', 'F7', 'AF3'],
        'Frontal Right (NE)': ['Fp2', 'F4', 'F8', 'AF4'],
        'Temporal Left': ['T7', 'P7', 'TP7'],
        'Temporal Right': ['T8', 'P8', 'TP8'],
        'Parietal (Top)': ['Pz', 'P3', 'P4'],
        'Central': ['Cz', 'FCz', 'CPz']
    }

    times_to_measure = [0.1, 0.3, 0.4, 0.5, 0.6]

    for t in times_to_measure:
        t_min, t_max = t - 0.025, t + 0.025
        for name, chans in rois.items():
            existing_chans = [c for c in chans if c in evoked_social.ch_names]
            if existing_chans:
                s_muv = evoked_social.copy().crop(t_min, t_max).pick(existing_chans).data.mean() * 1e6
                n_muv = evoked_numeric.copy().crop(t_min, t_max).pick(existing_chans).data.mean() * 1e6
                diff = s_muv - n_muv
                print(f"{name:20} | {t:5.1f}s | {s_muv:7.2f} µV | {n_muv:7.2f} µV | {diff:7.2f} µV")
        print(f"{'-'*75}")

    # 3. Rensa bort EXG-kanaler
    for ev in [evoked_social, evoked_numeric, evoked_diff]:
        ev.pick_types(eeg=True, exclude=['EXG7', 'EXG8'])

    # 4. Plotta Topomaps
    times = [0.1, 0.3, 0.4, 0.5, 0.6]
    my_vlim = vlim 

    # Plot Social
    fig1 = evoked_social.plot_topomap(times=times, vlim=my_vlim)
    fig1.suptitle(f"Average: Social {title_suffix} (Fixed Scale)")

    # Plot Numeric
    fig2 = evoked_numeric.plot_topomap(times=times, vlim=my_vlim)
    fig2.suptitle(f"Average: Numeric {title_suffix} (Fixed Scale)")

    # Plot Difference
    fig3 = evoked_diff.plot_topomap(times=times)
    fig3.suptitle(f"Difference: Social - Numeric {title_suffix}")

    # 5. Plotta ERP
    mne.viz.plot_compare_evokeds(
        dict(Social=evoked_social, Numeric=evoked_numeric),
        picks=['P7'], 
        title=f"ERP Comparison at P7 {title_suffix}"
    )
    
    plt.show() # Skriptet pausar här tills fönstren stängs

if __name__ == "__main__":
    root_folder = Path(__file__).resolve().parent
    proc_dir = root_folder / "EEG-proc"
    files = sorted(list(proc_dir.glob("sub-*_cleaned-epo.fif")))

    if not files:
        print(f"Hittade inga filer i {proc_dir}!")
        exit()

    print("\n--- EEG VISUALIZATION MENU ---")
    for i, f in enumerate(files):
        print(f"[{i+1}] {f.name}")
    print(f"[{len(files)+1}] GRAND AVERAGE (Combine all subs)")
    print(f"[{len(files)+2}] ALL (Cycle through individuals + Grand Average)")
    
    choice = input("\nVälj ett nummer: ").strip().lower()

    # Fixed scale to make comparisons meaningful
    fixed_vlim = (-10, 5)

    try:
        # Option: GRAND AVERAGE ONLY
        if choice == str(len(files)+1):
            print("\nGenerating GRAND AVERAGE only...")
            all_data = [mne.read_epochs(f, preload=True, verbose=False) for f in files]
            combined_epochs = mne.concatenate_epochs(all_data)
            run_visualization(combined_epochs, title_suffix="(Grand Average: All)", vlim=fixed_vlim)

        # Option: ALL (Individuals + Grand Average)
        elif choice == 'all' or choice == str(len(files)+2):
            all_epochs_list = []
            for f_path in files:
                print(f"\nProcessing individual: {f_path.name}")
                epochs = mne.read_epochs(f_path, preload=True, verbose=False)
                all_epochs_list.append(epochs)
                run_visualization(epochs, title_suffix=f"(Subject: {f_path.name[:6]})", vlim=fixed_vlim)
                time.sleep(0.2)
            
            print("\nGenerating GRAND AVERAGE for all subjects...")
            combined_epochs = mne.concatenate_epochs(all_epochs_list)
            run_visualization(combined_epochs, title_suffix="(Grand Average: All)", vlim=fixed_vlim)
        
        # Option: SINGLE SUBJECT
        else:
            idx = int(choice) - 1
            f = files[idx]
            epochs = mne.read_epochs(f, preload=True, verbose=False)
            run_visualization(epochs, title_suffix=f"(Subject: {f.name[:6]})", vlim=fixed_vlim)
            
    except Exception as e:
        print(f"Ett fel uppstod: {e}")