# EEG Inner Speech Recognition

This project provides a pipeline to preprocess raw EEG data, visualize evoked responses, and run a baseline inner-speech classifier using the OpenNeuro dataset `ds004197`.

The folder `Inner_Speech_EEG_fMRI/` contains the original reference material from the LTU repository. Our main workflow uses:

- `Inner_Speech_EEG_fMRI/EEG_preprocessing/preprocess.py`
- `visualize_eeg.py`
- `classify_eeg.py`

## Quick Start Guide

### 1. Local Setup

Raw data and processed files are not stored in Git, so each group member needs to create the local data folders manually.

Expected structure:

```text
EEG-project/
├── data/
│   ├── sub-01/
│   ├── sub-02/
│   ├── sub-03/
│   └── ...
├── EEG-proc/
├── classify_eeg.py
├── visualize_eeg.py
└── Inner_Speech_EEG_fMRI/
```

After downloading subject 1 EEG data, this file should exist:

```text
data/sub-01/ses-EEG/eeg/sub-01_ses-EEG_task-inner_eeg.bdf
```

The folder `EEG-proc/` is used for cleaned output files such as:

```text
EEG-proc/sub-01_cleaned-epo.fif
```

### 2. Download Data

Download dataset `ds004197` from OpenNeuro:

https://openneuro.org/datasets/ds004197

Place the downloaded subject folders inside `data/`.

### 3. Install Requirements

```bash
pip install mne numpy pandas scikit-learn matplotlib
```

## Running the Pipeline

Run the scripts in the following order.

### Step 1: Preprocessing

Preprocess one subject and create cleaned EEG epochs:

```bash
python Inner_Speech_EEG_fMRI/EEG_preprocessing/preprocess.py --id 1
```

This reads:

```text
data/sub-01/ses-EEG/eeg/sub-01_ses-EEG_task-inner_eeg.bdf
```

and writes:

```text
EEG-proc/sub-01_cleaned-epo.fif
```

Change `--id` to `2`, `3`, or `5` for the other subjects currently supported by the script.

### Step 2: Visualizing Brain Activity

Plot ERP and topographic maps for a cleaned subject:

```bash
python visualize_eeg.py --id 1
```

This script loads the cleaned `.fif` file from `EEG-proc/` and compares the `Social` and `Numeric` conditions.

### Step 3: Baseline Classification

Run the baseline CSP + LDA classifier:

```bash
python classify_eeg.py --id 1
```

This uses the cleaned epochs to test whether the model can separate social and numeric word conditions.

## Project Files

- `Inner_Speech_EEG_fMRI/EEG_preprocessing/preprocess.py`
  Main preprocessing script: trigger extraction, filtering, ICA, and epoching.
- `visualize_eeg.py`
  ERP and topomap visualization for cleaned EEG epochs.
- `classify_eeg.py`
  Baseline decoding script using CSP and LDA.
- `Inner_Speech_EEG_fMRI/EEG_preprocessing/processing_pipeline.m`
  Original MATLAB/EEGLAB preprocessing reference from the dataset authors.
- `Inner_Speech_EEG_fMRI/fMRI_preprocessing/`
  Reference files for the original fMRI preprocessing workflow.

## Notes

- The main scripts use the repository root together with the `data/` and `EEG-proc/` folders.
- The optional `--root` argument can be used if the project data is stored somewhere else.
- The MATLAB and fMRI files are included as reference material and are not required for the Python EEG pipeline above.
