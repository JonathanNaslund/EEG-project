import os
import numpy as np
import mne
from mne.io import read_raw_bdf
from pathlib import Path

# BioSemi-specific channels for eye activity (EOG)
eog_channels = ['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6']

class EEGPreprocess:
    def __init__(self, subject_id, root_path):
        self.subject_id = subject_id
        self.root_path = root_path
        sub = f'sub-0{subject_id:01d}'
        
        self.data_path = os.path.join(
            root_path, 'data', sub, 'ses-EEG', 'eeg',
            f'{sub}_ses-EEG_task-inner_eeg.bdf'
        )

    def run(self):
        print(f"\n" + "="*60)
        print(f" INITIALIZING PREPROCESSING FOR SUBJECT: {self.subject_id}")
        print(f"="*60)

        if not os.path.exists(self.data_path):
            print(f"[-] ERROR: Data file not found at: {self.data_path}")
            return False

        # 1. Load raw data
        print(f"[1/7] Loading Raw BDF Data...")
        raw = read_raw_bdf(self.data_path, preload=True, eog=eog_channels, stim_channel='Status', verbose=False)
        sfreq = raw.info['sfreq']
        n_channels = len(raw.ch_names)
        duration = raw.times[-1]
        print(f"      > Sampling Rate: {sfreq} Hz")
        print(f"      > Total Channels: {n_channels} ({raw.info['nchan']} signals)")
        print(f"      > Recording Duration: {duration:.2f} seconds")

        # 2. Find events with BioSemi mask
        print(f"[2/7] Extracting Triggers from 'Status' channel...")
        all_events = mne.find_events(raw, stim_channel='Status', shortest_event=1, mask=255, mask_type='and', verbose=False)
        valid_ids = [1, 2, 111, 112, 113, 114, 125, 126, 127, 128]
        mne_events = all_events[np.isin(all_events[:, 2], valid_ids)]
        
        unique, counts = np.unique(mne_events[:, 2], return_counts=True)
        print(f"      > Total raw events found: {len(all_events)}")
        print(f"      > Filtered valid events: {len(mne_events)}")
        for val, count in zip(unique, counts):
            print(f"        - Event ID {val}: {count} occurrences")

        # 3. Filtering
        l_freq, h_freq = 1.0, 40.0
        print(f"[3/7] Applying Band-pass Filter ({l_freq}Hz - {h_freq}Hz)...")
        raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', verbose=False)
        
        print(f"      > Applying Notch Filter at 50Hz (Line Noise)...")
        raw.notch_filter(freqs=50.0, verbose=False)

        # 4. Set electrode montage
        print(f"[4/7] Applying standard 'biosemi64' montage...")
        montage = mne.channels.make_standard_montage('biosemi64')
        raw.set_montage(montage, on_missing='ignore')
        print(f"      > 3D Digitized points applied to EEG channels.")

        # 5. ICA (Cleaning)
        n_comps = 20
        print(f"[5/7] Initializing ICA (Method: FastICA, Components: {n_comps})...")
        ica = mne.preprocessing.ICA(n_components=n_comps, random_state=42, method='fastica')
        ica.fit(raw, picks='eeg', verbose=False)
        
        print(f"      > Scanning for EOG artifacts using 'EXG1' reference...")
        eog_indices, _ = ica.find_bads_eog(raw, ch_name='EXG1', verbose=False)
        
        if len(eog_indices) > 0:
            ica.exclude = eog_indices
            print(f"      > Identified and excluded {len(eog_indices)} EOG components: {eog_indices}")
        else:
            ica.exclude = [0, 1]
            print(f"      > No automated EOG matches. Excluding default components [0, 1] as fallback.")
            
        ica.apply(raw, verbose=False)

        # 6. Epoching
        event_id_map = {
            'Fixation': 1, 'Rest': 2,
            'Social/Child': 111, 'Social/Daughter': 112, 'Social/Father': 113, 'Social/Wife': 114,
            'Numeric/Ten': 125, 'Numeric/Three': 126, 'Numeric/Six': 127, 'Numeric/Four': 128
        }
        tmin, tmax = -0.2, 0.8
        print(f"[6/7] Segmenting data into Epochs ({tmin}s to {tmax}s)...")
        epochs = mne.Epochs(
            raw, mne_events, event_id=event_id_map, tmin=tmin, tmax=tmax,
            baseline=(None, 0), preload=True, picks='eeg', 
            reject_by_annotation=False, proj=False, verbose=False
        )
        print(f"      > Epoching complete: {len(epochs)} segments created.")
        print(f"      > Data shape: {epochs.get_data().shape} (Epochs, Channels, Times)")

        # 7. Save results
        output_dir = os.path.join(self.root_path, 'EEG-proc')
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, f"sub-0{self.subject_id}_cleaned-epo.fif")
        
        print(f"[7/7] Saving processed data to disk...")
        epochs.save(out_file, overwrite=True, verbose=False)
        print(f"\n[!] SUCCESS: Subject {self.subject_id} processing complete.")
        print(f"    File saved: {out_file}")
        print("-" * 60)
        return True

if __name__ == "__main__":
    root_folder = Path(__file__).resolve().parent.parent.parent
    
    print("\n" + "#"*40)
    print("   EEG BATCH PREPROCESSING SYSTEM   ")
    print("#"*40)
    print(f"Project Root: {root_folder}")
    print("Available IDs: 1, 2, 3, 5")
    
    choice = input("\nEnter ID (e.g. '1'), list (e.g. '1,2,5'), or 'all': ").strip().lower()

    available_subjects = [1, 2, 3, 5]
    if choice == 'all':
        subjects_to_run = available_subjects
    else:
        try:
            subjects_to_run = [int(x.strip()) for x in choice.split(',')]
            subjects_to_run = [s for s in subjects_to_run if s in available_subjects]
        except ValueError:
            print("Invalid input. Please use numbers separated by commas.")
            subjects_to_run = []

    if not subjects_to_run:
        print("No valid subjects selected. Exiting.")
    else:
        for s_id in subjects_to_run:
            process = EEGPreprocess(subject_id=s_id, root_path=root_folder)
            process.run()
    
    print("\nProcessing session finished.")