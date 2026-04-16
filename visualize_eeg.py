import mne
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 1. Load the file
root_folder = Path(__file__).resolve().parent.parent
file_path = root_folder / "EEG-proc" / "sub-03_cleaned-epo.fif"
epochs = mne.read_epochs(file_path, preload=True)

# 2. Update event_id manually
# Here we map the names to the exact IDs that your log showed (111, 112... 128)
new_event_id = {
    'Social/Child': 111, 'Social/Daughter': 112, 
    'Social/Father': 113, 'Social/Wife': 114,
    'Numeric/Ten': 125, 'Numeric/Three': 126, 
    'Numeric/Six': 127, 'Numeric/Four': 128
}

# We need to filter the list so we only include IDs that actually exist in the file
actual_events = np.unique(epochs.events[:, 2])
filtered_event_id = {k: v for k, v in new_event_id.items() if v in actual_events}
epochs.event_id = filtered_event_id

print(f"Matched categories: {epochs.event_id}")

# 3. Check if we have data for the groups
if 'Social' in epochs.event_id.keys() or any(k.startswith('Social/') for k in epochs.event_id):
    print("Creating ERP for social words...")
    # Note: "/" in the name allows us to use just 'Social' to get all subcategories
    evoked_social = epochs['Social'].average()
    evoked_numeric = epochs['Numeric'].average()

    # --- Visualization 1: ERP curves ---
    # Compare Social vs Numeric at electrode Cz (center of the head)
    mne.viz.plot_compare_evokeds(
        dict(Social=evoked_social, Numeric=evoked_numeric),
        picks='Cz',
        title="Inner speech: Social vs Numeric words (Cz)"
    )

    # --- Visualization 2: Topomaps ---
    # See where in the brain activity occurs at 300 ms and 500 ms
    evoked_social_eeg = evoked_social.copy().pick_types(eeg=True, exclude=['EXG7', 'EXG8'])
    fig = evoked_social_eeg.plot_topomap(times=[0.3, 0.5])
    fig.suptitle("Topography: Social words")
    plt.show()
else:
    print("Could not find any events matching 'Social'.")
    print(f"IDs present in the file: {actual_events}")