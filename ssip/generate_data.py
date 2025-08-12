import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import numpy.matlib as matlib
import scipy, os, torch, glob 
import scipy.signal
import scipy.stats
import scipy.io.wavfile
import scipy.fftpack
import torch, torchaudio
from scipy.fft import fft, fftfreq
from pynwb import NWBHDF5IO
from scipy.io import wavfile
from scipy.signal import lfilter

hilbert3 = lambda x: scipy.signal.hilbert(x, scipy.fftpack.next_fast_len(len(x)),axis=0)[:len(x)]

def extractHG_modified(data, sr, windowLength=0.05, frameshift=0.01):
    #Linear detrend
    orig_data = scipy.signal.detrend(data, axis=0)
    #Filter High-Gamma Band
    sos = scipy.signal.iirfilter(4, [70/(sr/2),170/(sr/2)],btype='bandpass',output='sos')
    data = scipy.signal.sosfiltfilt(sos,orig_data,axis=0)
    #Attenuate first harmonic of line noise
    sos = scipy.signal.iirfilter(4, [98/(sr/2),102/(sr/2)],btype='bandstop',output='sos')
    data = scipy.signal.sosfiltfilt(sos,data,axis=0)
    # Attenuate second harmonic of line noise 
    sos = scipy.signal.iirfilter(4, [148/(sr/2),152/(sr/2)],btype='bandstop',output='sos')
    data = scipy.signal.sosfiltfilt(sos,data,axis=0)
    data = np.abs(hilbert3(data)) # this is the envelop amplitude
    # lowpass filter original signal to get the low-frequency signals (supposedly they are important for synthesis)
    sos = scipy.signal.iirfilter(6, 98/(sr/2), btype='lowpass', output='sos')
    data_extra = scipy.signal.sosfiltfilt(sos, orig_data, axis = 0)
    concatenated_data = np.concatenate([data, data_extra], axis = -1)
    return concatenated_data


def save_eeg_mel_pairs(eeg_chunks, log_mel_chunks, save_dir):
    os.makedirs(os.path.join(save_dir, "eeg"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "log_mel"), exist_ok=True)
    for idx, (eeg, log_mel) in enumerate(zip(eeg_chunks, log_mel_chunks)):
        # Save EEG
        eeg_path = os.path.join(save_dir, "eeg", f"{idx:03d}.npy")
        np.save(eeg_path, eeg.astype(np.float32))
        # Save mel
        log_mel_path = os.path.join(save_dir, "log_mel", f"{idx:03d}.npy")
        np.save(log_mel_path, log_mel)
    print(f"Saved {len(eeg_chunks)} EEG–log_mel pairs to {save_dir}")

def whisper_log_mel_spectrogram(audio: torch.Tensor, n_mels: int = 80) -> torch.Tensor:
    if audio.dim() == 2:
        audio = audio.mean(dim=0)  # convert stereo → mono
    elif audio.dim() != 1:
        raise ValueError("Audio tensor must be 1D or 2D (channels, samples)")
    sample_rate = 16000
    n_fft = 400
    hop_length = 160
    win_length = 400
    f_min = 0
    f_max = 8000
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window_fn=torch.hann_window,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm=None,
        mel_scale="htk"
    )
    mel = mel_transform(audio)
    log_mel = torch.clamp(mel, min=1e-10).log10()
    log_mel_numpy = log_mel.numpy() 
    return log_mel_numpy


if __name__ == "__main__":
    root_dir = '/depot/jgmakin/data/speech_decoding/Dutch_Dataset_processed'
    winL = 0.05
    frameshift = 0.01
    MIDDLE_ONLY = False  
    NUM_PAST_WINDOWS = 10
    path_bids = '/depot/jgmakin/data/speech_decoding/Dutch_Dataset'
    MAX_WAV_VALUES  = []
    file_list = [f"/depot/jgmakin/data/speech_decoding/Dutch_Dataset/sub-{i:02d}/ieeg/sub-{i:02d}_task-wordProduction_events.tsv" 
                    for i in range(1, 11)]

    participants = pd.read_csv(os.path.join(path_bids,'participants.tsv'), delimiter='\t')

    # ===== Step 1: Extract the EEG and AUDIO from the .nwb files =====
    for p_id, participant in enumerate(participants['participant_id']): 
        if p_id not in [2]:
            continue 
        io = NWBHDF5IO(os.path.join(path_bids,participant,'ieeg',f'{participant}_task-wordProduction_ieeg.nwb'), 'r')
        nwbfile = io.read()
        eeg = nwbfile.acquisition['iEEG'].data[:]; eeg_sr = 1024
        eeg_duration = eeg.shape[0]/eeg_sr
        print(f'sEEG length in seconds: {eeg_duration}')
        audio = nwbfile.acquisition['Audio'].data[:]; orig_audio_sr = 48000 
        audio_sr = 16000 # for Whisper
        audio = scipy.signal.resample(audio, int(len(audio)*audio_sr/orig_audio_sr))

        audio_duration = audio.shape[0]/audio_sr
        print(f'audio length in seconds: {audio_duration}')
        io.close()
        min_duration = min(audio_duration, eeg_duration)
        audio = audio[:int(min_duration*audio_sr)] ## mono audio
        eeg = eeg[:int(min_duration*eeg_sr), :]

        #Extract HG features
        feat = extractHG_modified(eeg, eeg_sr, windowLength=winL,frameshift=frameshift)
        print(f'eeg feature shape: {feat.shape}')  

    # ===== Step 2: Define parameters for (X,y) pair generation =====
    chunk_duration = 2  # seconds
    step_size = 3       # start every 3 sec → gaps between chunks
    pairs = []          # will store (eeg_chunk, audio_chunk)

    # ===== Step 3: Loop over eeg/audio chunks =====
    total_duration_sec = len(audio) / audio_sr
    start_times = np.arange(0, total_duration_sec - chunk_duration + 1, step_size)

    for start_sec in start_times:
        # Audio slicing
        start_a = int(start_sec * audio_sr)
        end_a = int((start_sec + chunk_duration) * audio_sr)
        audio_chunk = audio[start_a:end_a]
        audio_tensor = torch.from_numpy(audio_chunk.astype(np.float32))
        log_mel_chunk = whisper_log_mel_spectrogram(audio_tensor)
        # EEG slicing
        start_e = int(start_sec * eeg_sr)
        end_e = int((start_sec + chunk_duration) * eeg_sr)
        eeg_chunk = eeg[start_e:end_e, :]
        pairs.append((eeg_chunk, log_mel_chunk))

    # ===== Step 4: Example: convert to dataset arrays =====
    eeg_chunks = [p[0] for p in pairs]  # list of (T_eeg, channels)
    log_mel_chunks = [p[1] for p in pairs]  # list of (??)

    print(f"Created {len(pairs)} EEG→audio pairs.")
    print(f"First EEG chunk shape: {eeg_chunks[0].shape}")
    print(f"First log_mel chunk shape: {log_mel_chunks[0].shape}")

    save_eeg_mel_pairs(eeg_chunks, log_mel_chunks, root_dir)



