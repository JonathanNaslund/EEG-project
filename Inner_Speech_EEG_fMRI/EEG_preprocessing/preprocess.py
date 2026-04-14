import os
import numpy as np
import pandas as pd
import argparse
import mne
from mne.io import read_raw_bdf

# BioSemi-specifika kanaler för referens och ögon (EOG)
eog_channels = ['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6']

class EEGPreprocess:
    def __init__(self, subject_id, root_path):
        self.subject_id = subject_id
        self.root_path = root_path
        # Korrekt mappstruktur enligt ditt dataset (sub-01/ses-EEG/eeg/...)
        self.data_path = os.path.join(
            root_path, 'data', f'sub-0{subject_id}', 'ses-EEG', 'eeg', 
            f'sub-0{subject_id}_ses-EEG_task-inner_eeg.bdf'
        )

    def run(self):
        print(f"\n--- Startar bearbetning för subjekt {self.subject_id} ---")
        
        if not os.path.exists(self.data_path):
            print(f"FEL: Hittade inte filen {self.data_path}")
            return

        # 1. Läs in rådata (Här definierar vi Status som stim_channel)
        print("Läser in rådata...")
        raw = read_raw_bdf(self.data_path, preload=True, eog=eog_channels, stim_channel='Status')
        
       # 2. Hitta events direkt i Status-kanalen
        print("Söker efter triggers i Status-kanalen...")
        
        # BioSemi skickar ofta triggers på de lägre 8 bitarna (0-255).
        # Vi använder en mask (255) för att ignorera system-signaler som 65536.
        all_events = mne.find_events(raw, stim_channel='Status', 
                                   shortest_event=1, 
                                   mask=255,           # <--- DENNA ÄR KRITISKT VIKTIG
                                   mask_type='and')    # <--- OCH DENNA
        
        # Uppdaterade ID:n för att matcha din logg
        # 1: Fixation, 2: Vila
        # 111-114: Sociala ord, 125-128: Numeriska ord
        valid_ids = [1, 2, 111, 112, 113, 114, 125, 126, 127, 128]
        mne_events = all_events[np.isin(all_events[:, 2], valid_ids)]
        
        print(f"Hittade {len(mne_events)} giltiga triggers.")
        print(f"Unika ID:n som faktiskt behålls: {np.unique(mne_events[:, 2])}")

        # 3. Filtrering
        print("Filtrerar (1-40 Hz) och tar bort 50Hz brus...")
        raw.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin')
        raw.notch_filter(freqs=50.0)

        # 4. Sätt elektrod-montage (64-kanal BioSemi)
        montage = mne.channels.make_standard_montage('biosemi64')
        raw.set_montage(montage, on_missing='ignore')

        # 5. ICA (Städa bort ögonblinkningar)
        print("Kör ICA (Artefakt-reducering)...")
        ica = mne.preprocessing.ICA(n_components=20, random_state=42, method='fastica')
        ica.fit(raw, picks='eeg')
        
        # Försök hitta blinkningar automatiskt via EXG1
        eog_indices, _ = ica.find_bads_eog(raw, ch_name='EXG1', verbose=False)
        
        # Om auto-find misslyckas, ta bort de två första komponenterna (oftast ögon)
        if len(eog_indices) == 0:
            print("Auto-ICA hittade inget. Exkluderar komponent 0 och 1 (standard för ögon/blink).")
            ica.exclude = [0, 1]
        else:
            print(f"ICA hittade {len(eog_indices)} artefakt-komponenter.")
            ica.exclude = eog_indices
            
        ica.apply(raw)

        # 6. Epokning (Klipp ut tanke-segmenten)
        print("Skapar epoker...")
        tmin, tmax = -0.2, 0.8
        
        epochs = mne.Epochs(
            raw, 
            mne_events, 
            event_id=None, 
            tmin=tmin, 
            tmax=tmax, 
            baseline=(None, 0), 
            preload=True, 
            picks='eeg',
            reject=None,   # Kasta inte data pga hög amplitud
            flat=None,     # Kasta inte data pga låg amplitud
            reject_by_annotation=False, # Ignorera interna bad-markeringar
            proj=False
        )

        print(f"KLAR! Antal sparade segment: {len(epochs)}")
        
        # 7. Spara resultatet till EEG-proc mappen
        output_dir = os.path.join(self.root_path, 'EEG-proc')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        out_file = os.path.join(output_dir, f"sub-0{self.subject_id}_cleaned-epo.fif")
        epochs.save(out_file, overwrite=True)
        print(f"Filen sparad: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=1, choices=[1,2,3,5], help="Subject ID")
    args = parser.parse_args()

    # Din root-mapp
    root_folder = "C:/projects/brain_imaging"
    
    process = EEGPreprocess(subject_id=args.id, root_path=root_folder)
    process.run()