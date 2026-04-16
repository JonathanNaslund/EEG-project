import os
import numpy as np
import argparse
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
            root_path,
            'data',
            sub,
            'ses-EEG',
            'eeg',
            f'{sub}_ses-EEG_task-inner_eeg.bdf'
        )

    def run(self):
        print(f"\n--- Starting processing for subject {self.subject_id} ---")

        if not os.path.exists(self.data_path):
            print(f"ERROR: Could not find file {self.data_path}")
            return

        # 1. Load raw data
        print("Loading raw data...")
        raw = read_raw_bdf(self.data_path, preload=True, eog=eog_channels, stim_channel='Status')

        # 2. Find events directly in the Status channel
        print("Searching for triggers in the Status channel...")
        all_events = mne.find_events(
            raw,
            stim_channel='Status',
            shortest_event=1,
            mask=255,
            mask_type='and'
        )

        valid_ids = [1, 2, 111, 112, 113, 114, 125, 126, 127, 128]
        mne_events = all_events[np.isin(all_events[:, 2], valid_ids)]

        print(f"Found {len(mne_events)} valid triggers.")
        print(f"Unique IDs actually kept: {np.unique(mne_events[:, 2])}")

        # 3. Filtering
        print("Filtering (0.3-40 Hz) and removing 50 Hz noise...")
        raw.filter(l_freq=0.3, h_freq=40.0, fir_design='firwin')
        raw.notch_filter(freqs=50.0)

        # 4. Set electrode montage
        montage = mne.channels.make_standard_montage('biosemi64')
        raw.set_montage(montage, on_missing='ignore')

        # 5. Explicit EEG reference
        print("Applying average reference...")
        raw.set_eeg_reference('average', projection=False)

        # 6. ICA
        print("Running ICA (artifact reduction)...")
        ica = mne.preprocessing.ICA(n_components=20, random_state=42, method='fastica')
        ica.fit(raw, picks='eeg')

        found_eog_components = set()
        for ch in ['EXG1', 'EXG2', 'EXG3', 'EXG4']:
            if ch in raw.ch_names:
                eog_indices, _ = ica.find_bads_eog(raw, ch_name=ch, verbose=False)
                found_eog_components.update(eog_indices)

        found_eog_components = sorted(list(found_eog_components))

        if len(found_eog_components) == 0:
            print("Auto-ICA found no clear EOG-related components. No ICA components will be excluded.")
            ica.exclude = []
        else:
            print(f"ICA found {len(found_eog_components)} artifact components.")
            ica.exclude = found_eog_components

        ica.apply(raw)

        # 7. Epoching
        print("Creating epochs...")
        tmin, tmax = -0.2, 0.8

        event_id_map = {
            'Fixation': 1,
            'Rest': 2,
            'Social/Child': 111,
            'Social/Daughter': 112,
            'Social/Father': 113,
            'Social/Wife': 114,
            'Numeric/Ten': 125,
            'Numeric/Three': 126,
            'Numeric/Six': 127,
            'Numeric/Four': 128
        }

        epochs = mne.Epochs(
            raw,
            mne_events,
            event_id=event_id_map,
            tmin=tmin,
            tmax=tmax,
            baseline=(None, 0),
            preload=True,
            picks='eeg',
            reject=None,
            flat=None,
            reject_by_annotation=True,
            proj=False,
            verbose=False
        )

        print(f"DONE! Number of saved segments: {len(epochs)}")

        # 8. Save results
        output_dir = os.path.join(self.root_path, 'EEG-proc')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        out_file = os.path.join(output_dir, f"sub-0{self.subject_id}_cleaned-epo.fif")
        epochs.save(out_file, overwrite=True)
        print(f"File saved: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=1, choices=[1, 2, 3, 5], help="Subject ID")
    args = parser.parse_args()

    root_folder = Path(__file__).resolve().parent.parent.parent.parent

    process = EEGPreprocess(subject_id=args.id, root_path=root_folder)
    process.run()