import mne
import matplotlib.pyplot as plt
import numpy as np

# 1. Ladda filen
file_path = "C:/projects/brain_imaging/EEG-proc/sub-01_cleaned-epo.fif"
epochs = mne.read_epochs(file_path, preload=True)

# 2. Uppdatera event_id manuellt
# Här mappar vi namnen till de exakta ID:n som din logg visade (111, 112... 128)
new_event_id = {
    'Social/Child': 111, 'Social/Daughter': 112, 
    'Social/Father': 113, 'Social/Wife': 114,
    'Numeric/Ten': 125, 'Numeric/Three': 126, 
    'Numeric/Six': 127, 'Numeric/Four': 128
}

# Vi måste filtrera listan så vi bara lägger till de ID:n som faktiskt finns i filen
actual_events = np.unique(epochs.events[:, 2])
filtered_event_id = {k: v for k, v in new_event_id.items() if v in actual_events}
epochs.event_id = filtered_event_id

print(f"Matchade kategorier: {epochs.event_id}")

# 3. Kontrollera om vi har data för grupperna
if 'Social' in epochs.event_id.keys() or any(k.startswith('Social/') for k in epochs.event_id):
    print("Skapar ERP för Sociala ord...")
    # Notera: "/" i namnet gör att vi kan skriva bara 'Social' för att få alla underkategorier
    evoked_social = epochs['Social'].average()
    evoked_numeric = epochs['Numeric'].average()

    # --- Visualisering 1: ERP Kurvor ---
    # Jämför Sociala vs Numeriska på elektrod Cz (mitt på huvudet)
    mne.viz.plot_compare_evokeds(
        dict(Social=evoked_social, Numeric=evoked_numeric),
        picks='Cz',
        title="Inre tal: Sociala vs Numeriska ord (Cz)"
    )

    # --- Visualisering 2: Topomaps ---
    # Se var i hjärnan det händer saker vid 300ms och 500ms
    evoked_social.plot_topomap(times=[0.3, 0.5], title="Topografi: Sociala ord")
    
    plt.show()
else:
    print("Kunde inte hitta några events som matchar 'Social'.")
    print(f"ID:n som finns i filen: {actual_events}")