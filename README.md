# EEG Inner Speech Project

This repository is our group workspace for preprocessing EEG data from the OpenNeuro dataset `ds004197`, visualizing the cleaned signal, and running a simple baseline classifier for inner-speech decoding.

The original research material from the LTU repository is kept in `Inner_Speech_EEG_fMRI/`. Our group mainly uses the Python scripts in this repo root plus the EEG preprocessing script in `Inner_Speech_EEG_fMRI/EEG_preprocessing/`.

## What Goes Where

Keep the Git repository itself in one folder, and put the downloaded OpenNeuro dataset inside the repo in a folder called `data/`.

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
    └── EEG_preprocessing/
        └── preprocess.py
```

For example, after downloading subject 1 EEG data, this file should exist:

```text
data/sub-01/ses-EEG/eeg/sub-01_ses-EEG_task-inner_eeg.bdf
```

The `EEG-proc/` folder will be created automatically when preprocessing runs. It stores cleaned epoch files such as:

```text
EEG-proc/sub-01_cleaned-epo.fif
```

## What Each Script Does

- `Inner_Speech_EEG_fMRI/EEG_preprocessing/preprocess.py`
  Reads raw `.bdf` EEG data, finds triggers, filters the signal, runs ICA, creates epochs, and saves cleaned `.fif` files.
- `visualize_eeg.py`
  Loads a cleaned `.fif` file and plots ERP comparisons and topomaps for social vs numeric words.
- `classify_eeg.py`
  Loads a cleaned `.fif` file and runs a baseline CSP + LDA classifier.
- `Inner_Speech_EEG_fMRI/EEG_preprocessing/processing_pipeline.m`
  Original MATLAB/EEGLAB reference pipeline from the dataset authors. This is reference material, not part of our main Python workflow.

## Setup

Install the Python packages used by the current scripts:

```bash
pip install mne numpy pandas scikit-learn matplotlib
```

Download the dataset from OpenNeuro:

https://openneuro.org/datasets/ds004197

Place the downloaded subject folders inside `data/` in this repository.

## Run Order

Run the scripts from the project root in this order.

### 1. Preprocess raw EEG

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

You can change `--id` to `2`, `3`, or `5`.

### 2. Visualize the cleaned EEG

```bash
python visualize_eeg.py --id 1
```

This opens ERP and topomap figures for the cleaned epochs of that subject.

### 3. Run baseline classification

```bash
python classify_eeg.py --id 1
```

This runs a simple binary classification between social and numeric word conditions and prints the cross-validation accuracy.

## Notes

- All main scripts now use project-relative paths, so they work on this Mac as long as you run them from this repository.
- If you want to use another project location, the scripts also support `--root` to point to a different project root containing `data/` and `EEG-proc/`.
- The fMRI files inside `Inner_Speech_EEG_fMRI/fMRI_preprocessing/` are reference scripts from the original dataset repository and are not required for the EEG pipeline above.
