import mne
import numpy as np
import os
import pandas as pd
import pickle
from src.config import ann2label


def filter_signal(raw, channel, lowcut=0.5, highcut=30.0):
    raw_filt = raw.copy()
    raw_filt.filter(
        l_freq=lowcut, 
        h_freq=highcut, 
        picks=channel,
        method='fir',
        phase='zero')
    return raw_filt

def fix_epoch_length(epoch_data, desired_length):
    current_length = epoch_data.shape[1]
    if current_length > desired_length:
        return epoch_data[:, :desired_length]
    elif current_length < desired_length:
        pad_width = desired_length - current_length
        last_segment = epoch_data[:, -min(current_length, pad_width):]
        repeated = np.tile(last_segment, int(np.ceil(pad_width / last_segment.shape[1])))
        repeated = repeated[:, :pad_width]
        epoch_mean = np.mean(epoch_data)
        mean_array = np.full((epoch_data.shape[0], pad_width), epoch_mean)
        pad_values = 0.5 * (repeated + mean_array)
        return np.concatenate((epoch_data, pad_values), axis=1)
    return epoch_data

def zscore_normalize(epoch):
    mean = np.mean(epoch)
    std = np.std(epoch)
    if std == 0:
        return epoch - mean
    return (epoch - mean) / std

# extract epochs theo label trong tsv
def extract_epochs(raw, ann_df, channel, sfreq, epoch_duration, ann2label):
    epochs = []
    labels = []
    onsets = []

    n_samples_per_epoch = int(epoch_duration * sfreq)

    valid_ann = ann_df[ann_df['description'].isin(ann2label.keys())]

    for _, row in valid_ann.iterrows():
        start_time = row['onset']
        duration = row['duration']
        label_str = row['description']
        
        if duration < epoch_duration - 0.5: 
            continue
            
        start_sample = int(start_time * sfreq)
        end_sample = start_sample + n_samples_per_epoch

        if end_sample > raw.n_times:
            continue

        ch_data = raw.get_data(picks=channel, start=start_sample, stop=end_sample)

        # Safety check
        ch_data = fix_epoch_length(ch_data, n_samples_per_epoch)
        ch_data = zscore_normalize(ch_data)

        epochs.append(ch_data)
        labels.append(ann2label[label_str])
        onsets.append(start_time)

    return epochs, labels, onsets

def run_pipeline(data_dir, output_dir, channel='EEG C3-M2', epoch_duration=30):
    os.makedirs(output_dir, exist_ok=True)
    edf_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".edf")])
    tsv_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".tsv")])
    
    for edf_file, tsv_file in zip(edf_files, tsv_files):
        study_id = edf_file.split("_")[0]
        output_path = os.path.join(output_dir, f"data_{study_id}.pkl")

        if os.path.exists(output_path):
            print(f"Skip {study_id}")
            continue
        print(f"Processing: {edf_file}...")
        raw = mne.io.read_raw_edf(os.path.join(data_dir, edf_file), preload=True)
        ann_df = pd.read_csv(os.path.join(data_dir, tsv_file), sep="\t")
        
        raw_filt = filter_signal(raw, channel)
        raw_filt.pick([channel])
        sfreq = raw.info['sfreq']
        
        target_fs = 256
        if sfreq != target_fs:
            print(f"  Resample {sfreq}Hz → {target_fs}Hz")
            raw_filt.resample(target_fs)
            sfreq = target_fs
        
        epochs, labels, onsets = extract_epochs(
            raw_filt, ann_df, channel, sfreq, epoch_duration, ann2label
        )
        
        if len(epochs) == 0:
            print(f"No valid epochs found for {study_id}")
            continue

        data_dict = {
            'x': np.array(epochs, dtype=np.float32).squeeze()[:, :, np.newaxis],
            'y': np.array(labels, dtype=np.int32), 
            'fs': sfreq,
            'channel': channel,
            'n_epochs': len(epochs),
            'onsets': np.array(onsets),
        }
        
        with open(output_path, "wb") as f:
            pickle.dump(data_dict, f)

        print(f"Saved {len(epochs)} epochs")
        
if __name__ == "__main__":
    run_pipeline("./data/raw", "./data/processed")