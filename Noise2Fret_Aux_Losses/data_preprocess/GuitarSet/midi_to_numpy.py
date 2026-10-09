"""
Code adapted from https://github.com/KimSehun725/Tab-estimator/tree/master
"""


import numpy as np
import yaml
import pretty_midi
import os
from scipy.io import wavfile
import librosa
import math


def pitch_to_nfrets(pitch, string_name):
    string_midi_pitches = [40, 45, 50, 55, 59, 64]
    base_pitch_key = {"E string": 40,
                      "A string": 45,
                      "D string": 50,
                      "G string": 55,
                      "B string": 59,
                      "e string": 64}

    string_n_key = {"E string": 0,
                    "A string": 1,
                    "D string": 2,
                    "G string": 3,
                    "B string": 4,
                    "e string": 5}
    base_pitch = base_pitch_key[string_name]
    string_n = string_n_key[string_name]
    return pitch - base_pitch, string_n

def process_stft_sf_b(data, sr_original, **kwargs):

    down_sampling_rate = kwargs["down_sampling_rate"]
    hop_length = 512
    n_fft = 1024

    data = data.astype(float)
    data = librosa.util.normalize(data)
    data = librosa.resample(y=data, orig_sr=sr_original, target_sr=down_sampling_rate)
    mag = np.abs(librosa.stft(y=data,
                               n_fft=n_fft,
                               hop_length=hop_length,
                               ))

    # --- Spectral Flux ---
    flux = np.diff(mag, axis=-1)  # (F, T-1)
    flux = np.maximum(flux, 0.0)  # half-wave rectify
    flux = flux.sum(axis=0)  # sum over freqs -> (T-1,)
    flux = np.pad(flux, (1, 0), mode='constant', constant_values=0.0)  # (T,)

    # --- Spectral Brightness ---
    cutoff_hz = 1500.0
    n_fft = (mag.shape[0] - 1) * 2  # infer n_fft from number of freq bins

    freqs = librosa.fft_frequencies(sr=down_sampling_rate, n_fft=n_fft)  # (F,)
    bright_mask = freqs >= cutoff_hz  # (F,) boolean

    energy_total = (mag ** 2).sum(axis=0)  # (T,)
    energy_bright = (mag[bright_mask, :] ** 2).sum(axis=0)  # (T,)
    brightness = energy_bright / np.maximum(energy_total, 1e-8)  # (T,)

    return mag, flux, brightness



def process_cqt(data, sr_original, **kwargs):

    down_sampling_rate = kwargs["down_sampling_rate"]
    bins_per_octave = kwargs["bins_per_octave"]
    n_bins = kwargs["n_bins"]
    hop_length = kwargs["hop_length"]

    data = data.astype(float)
    data = librosa.util.normalize(data)
    data = librosa.resample(y=data, orig_sr=sr_original, target_sr=down_sampling_rate)
    data = np.abs(librosa.cqt(data,
                              hop_length=hop_length,
                              sr=down_sampling_rate,
                              n_bins=n_bins,
                              bins_per_octave=bins_per_octave))
    return data


def process_mel_spec(data, sr_original, **kwargs):
    down_sampling_rate = kwargs["down_sampling_rate"]
    hop_length = kwargs["hop_length"]
    data = data.astype(float)
    data = librosa.util.normalize(data)
    data = librosa.resample(y=data, orig_sr=sr_original, target_sr=down_sampling_rate)
    data = np.abs(librosa.feature.melspectrogram(y=data,
                                                 sr=down_sampling_rate,
                                                 n_fft=2048,
                                                 hop_length=hop_length))
    return data


def main(midi_filename_list, kwargs):
    note_resolution = kwargs["note_resolution"]
    down_sampling_rate = kwargs["down_sampling_rate"]
    hop_length = kwargs["hop_length"]
    midi_filename_list = midi_filename_list.split("\n")

    if "auto_quantized" in midi_filename_list[0]:
        npz_dir = os.path.join(
            "data", "npz", f"auto_quantized_{note_resolution}")
        audio_dir = os.path.join("data", "wav", "auto_quantized_16")
    elif "original" in midi_filename_list[0]:
        npz_dir = os.path.join("data", "npz", "original")
        audio_dir = os.path.join("/Users/simionato/PycharmProjects/Tab-estimator/data/GuitarSet", "audio_mono-mic")
    else:
        print("error")

    norm_len = kwargs["down_sampling_rate"] / kwargs["hop_length"]

    for midi_filename in midi_filename_list:
        audio_filename = os.path.join(
            audio_dir, os.path.split(midi_filename)[1][:-4] + "_mic.wav")
        sr_original, audio_file = wavfile.read(audio_filename)
        cqt = process_cqt(audio_file, sr_original, **kwargs)
        log_cqt = librosa.amplitude_to_db(np.abs(cqt))
        stft, sf, b = process_stft_sf_b(audio_file, sr_original, **kwargs)
        log_stft = librosa.amplitude_to_db(stft)
        mel_spec = process_mel_spec(audio_file, sr_original, **kwargs)

        audio = audio_file.astype(float)
        audio = librosa.util.normalize(audio)
        audio = librosa.resample(y=audio, orig_sr=sr_original, target_sr=down_sampling_rate)

        cqt = cqt.T
        stft = stft.T
        b = b[:, None]
        sf = sf[:, None]
        log_cqt = log_cqt.T
        log_stft = log_stft.T
        mel_spec = mel_spec.T

        midi_file = pretty_midi.PrettyMIDI(midi_filename)
        npz_path = os.path.join(npz_dir, os.path.split(midi_filename)[1][:-4])
        tempo = float(midi_filename.split('-')[1])
        note_dur = 60 / tempo / note_resolution * 4

        len_in_notes = int(math.ceil(round(midi_file.get_end_time(
        ) / note_dur) / (note_resolution * 4)) * (note_resolution * 4))

        feature_len = int(len_in_notes * note_dur *
                          (down_sampling_rate / hop_length))
        cqt = cqt[:feature_len]
        mel_spec = mel_spec[:feature_len]
        stft = stft[:feature_len]
        sf = sf[:feature_len]
        b = b[:feature_len]

        # process note-wise tab
        tab = np.zeros((len_in_notes, 6, 21))
        tab[:, :, 20] = 1
        for midi_string in midi_file.instruments:
            string_name = midi_string.name
            for j, note in enumerate(midi_string.notes):
                frets, string_n = pitch_to_nfrets(note.pitch, string_name)
                tab[int(round(note.start / note_dur)) : int(round(note.end / note_dur)), string_n, 20] = 0
                tab[int(round(note.start / note_dur)) : int(round(note.end / note_dur)), string_n, frets] = 1

        # process note-wise tab onset
        tab_onset = np.zeros((len_in_notes, 6, 21))
        tab_onset[:, :, 20] = 1
        for midi_string in midi_file.instruments:
            string_name = midi_string.name
            for j, note in enumerate(midi_string.notes):
                frets, string_n = pitch_to_nfrets(note.pitch, string_name)
                if int(round(note.start / note_dur)) >= len_in_notes:
                    break
                tab_onset[int(round(note.start / note_dur)), string_n, 20] = 0
                tab_onset[int(round(note.start / note_dur)),
                          string_n, frets] = 1

        # process frame-wise tab
        frame_tab = np.zeros((feature_len, 6, 21))
        frame_tab[:, :, 20] = 1
        for midi_string in midi_file.instruments:
            string_name = midi_string.name
            for j, note in enumerate(midi_string.notes):
                frets, string_n = pitch_to_nfrets(note.pitch, string_name)
                frame_tab[int(round(note.start * norm_len)) : int(round(note.end * norm_len)), string_n, 20] = 0
                frame_tab[int(round(note.start * norm_len)) : int(round(note.end * norm_len)), string_n, frets] = 1

        # process frame-wise tab onset
        frame_tab_onset = np.zeros((feature_len, 6, 21))
        frame_tab_onset[:, :, 20] = 1
        for midi_string in midi_file.instruments:
            string_name = midi_string.name
            for j, note in enumerate(midi_string.notes):
                frets, string_n = pitch_to_nfrets(note.pitch, string_name)
                frame_tab_onset[int(
                    round(note.start * norm_len)), string_n, 20] = 0
                frame_tab_onset[int(round(note.start * norm_len)),
                                string_n, frets] = 1

        # process note-wise F0
        F0 = np.zeros((len_in_notes, 44))
        for midi_string in midi_file.instruments:
            string_name = midi_string.name
            for j, note in enumerate(midi_string.notes):
                pitch = note.pitch - 40
                F0[int(round(note.start / note_dur)) : int(round(note.end / note_dur)), pitch] = 1

        # process note-wise F0 onset
        F0_onset = np.zeros((len_in_notes, 44))
        for midi_string in midi_file.instruments:
            string_name = midi_string.name
            for j, note in enumerate(midi_string.notes):
                if int(round(note.start / note_dur)) >= len_in_notes:
                    break
                pitch = note.pitch - 40
                F0_onset[int(round(note.start / note_dur)), pitch] = 1

        # process frame-wise F0
        frame_F0 = np.zeros((feature_len, 44))
        for midi_string in midi_file.instruments:
            string_name = midi_string.name
            for j, note in enumerate(midi_string.notes):
                pitch = note.pitch - 40
                frame_F0[int(round(note.start * norm_len)) : int(round(note.end * norm_len)), pitch] = 1

        # process frame-wise F0 onset
        frame_F0_onset = np.zeros((feature_len, 44))
        for midi_string in midi_file.instruments:
            string_name = midi_string.name
            for j, note in enumerate(midi_string.notes):
                pitch = note.pitch - 40
                frame_F0_onset[int(round(note.start * norm_len)), pitch] = 1

        """
        # 0~19 : number of frets
        # 20 : not played

        """
        np.savez_compressed(npz_path,
                            cqt=cqt,
                            stft=stft,
                            sf=sf,
                            b=b,
                            audio=audio,
                            log_cqt=log_cqt,
                            mel_spec=mel_spec,
                            tab=tab,
                            tab_onset=tab_onset,
                            frame_tab=frame_tab,
                            frame_tab_onset=frame_tab_onset,
                            F0=F0,
                            F0_onset=F0_onset,
                            frame_F0=frame_F0,
                            frame_F0_onset=frame_F0_onset,
                            tempo=tempo,
                            len_in_notes=len_in_notes)
        split_save(npz_path, cqt, stft, sf, b, audio, mel_spec, tab, tab_onset, frame_tab, frame_tab_onset, F0, F0_onset,
                   frame_F0, frame_F0_onset,  tempo, note_resolution, hop_length)
        print("finished", os.path.split(npz_path)[1])


def split_save(npz_path, cqt, stft, sf, b, audio, mel_spec, tab, tab_onset, frame_tab, frame_tab_onset, F0, F0_onset, frame_F0, frame_F0_onset, tempo, note_resolution, hop):
    feature_len = cqt.shape[0]
    note_len = tab.shape[0]
    note_len_4bars = int(round(float(note_len) / float(note_resolution * 4)))
    feature_len_4bars = int(round(feature_len / note_len_4bars))
    pad_len = 0
    split_npz_path = os.path.join(os.path.split(npz_path)[0], "split")

    if not(os.path.exists(split_npz_path)):
        os.makedirs(split_npz_path)

    for n_4bars in range(note_len_4bars):
        split_npz_filename = os.path.join(
            split_npz_path, (os.path.split(npz_path)[1] + f"_0{n_4bars}"))

        split_tab = tab[(note_resolution * 4) *
                        n_4bars: (note_resolution * 4) * (n_4bars+1)]
        split_tab_onset = tab_onset[(note_resolution * 4) *
                                    n_4bars: (note_resolution * 4) * (n_4bars+1)]
        split_F0 = F0[(note_resolution * 4) *
                      n_4bars: (note_resolution * 4) * (n_4bars+1)]
        split_F0_onset = F0_onset[(note_resolution * 4)
                                  * n_4bars: (note_resolution * 4) * (n_4bars+1)]

        w_start = feature_len_4bars * n_4bars * hop - pad_len * hop
        w_end = feature_len_4bars * (n_4bars + 1) * hop + pad_len * hop

        cqt_zero_pad = np.zeros((pad_len, cqt.shape[1]))
        mel_spec_zero_pad = np.zeros((pad_len, mel_spec.shape[1]))
        stft_zero_pad = np.zeros((pad_len, stft.shape[1]))
        sf_zero_pad = np.zeros((pad_len, sf.shape[1]))
        b_zero_pad = np.zeros((pad_len, b.shape[1]))
        frame_F0_zero_pad = np.zeros((pad_len, frame_F0.shape[1]))
        frame_tab_zero_pad = np.zeros((pad_len, 6, 21))
        waveform_zero = np.zeros(pad_len * hop)

        if n_4bars == 0:
            split_waveform = np.append(waveform_zero, audio[0 : (feature_len_4bars + pad_len) * hop], axis=0)
            split_cqt = np.append(
                cqt_zero_pad, cqt[feature_len_4bars * n_4bars: feature_len_4bars * (n_4bars+1) + pad_len], axis=0)
            split_stft = np.append(
                stft_zero_pad, stft[feature_len_4bars * n_4bars: feature_len_4bars * (n_4bars + 1) + pad_len], axis=0)
            split_sf = np.append(
                sf_zero_pad, sf[feature_len_4bars * n_4bars: feature_len_4bars * (n_4bars + 1) + pad_len], axis=0)
            split_b = np.append(
                b_zero_pad, b[feature_len_4bars * n_4bars: feature_len_4bars * (n_4bars + 1) + pad_len], axis=0)
            split_mel_spec = np.append(
                mel_spec_zero_pad, mel_spec[feature_len_4bars * n_4bars: feature_len_4bars * (n_4bars+1) + pad_len], axis=0)
            split_frame_tab = np.append(
                frame_tab_zero_pad, frame_tab[feature_len_4bars * n_4bars: feature_len_4bars * (n_4bars+1) + pad_len], axis=0)
            split_frame_tab_onset = np.append(
                frame_tab_zero_pad, frame_tab_onset[feature_len_4bars * n_4bars: feature_len_4bars * (n_4bars+1) + pad_len], axis=0)
            split_frame_F0 = np.append(
                frame_F0_zero_pad, frame_F0[feature_len_4bars * n_4bars: feature_len_4bars * (n_4bars+1) + pad_len], axis=0)
            split_frame_F0_onset = np.append(
                frame_F0_zero_pad, frame_F0_onset[feature_len_4bars * n_4bars: feature_len_4bars * (n_4bars+1) + pad_len], axis=0)
        elif n_4bars == note_len_4bars-1:
            split_waveform = np.append(audio[w_start: feature_len_4bars * n_4bars * hop + feature_len_4bars * hop],
                                       waveform_zero, axis=0)
            split_cqt = np.append(
                cqt[feature_len_4bars * n_4bars - pad_len: feature_len_4bars * (n_4bars+1)], cqt_zero_pad, axis=0)
            split_stft = np.append(
                stft[feature_len_4bars * n_4bars - pad_len: feature_len_4bars * (n_4bars + 1)], stft_zero_pad, axis=0)
            split_sf = np.append(
                sf[feature_len_4bars * n_4bars - pad_len: feature_len_4bars * (n_4bars + 1)], sf_zero_pad, axis=0)
            split_b = np.append(
                b[feature_len_4bars * n_4bars - pad_len: feature_len_4bars * (n_4bars + 1)], b_zero_pad, axis=0)
            split_mel_spec = np.append(
                mel_spec[feature_len_4bars * n_4bars - pad_len: feature_len_4bars * (n_4bars+1)], mel_spec_zero_pad, axis=0)
            split_frame_tab = np.append(
                frame_tab[feature_len_4bars * n_4bars - pad_len: feature_len_4bars *
                          (n_4bars+1)], frame_tab_zero_pad, axis=0)
            split_frame_tab_onset = np.append(
                frame_tab_onset[feature_len_4bars * n_4bars - pad_len: feature_len_4bars *
                                (n_4bars+1)], frame_tab_zero_pad, axis=0)
            split_frame_F0 = np.append(
                frame_F0[feature_len_4bars * n_4bars - pad_len: feature_len_4bars *
                         (n_4bars+1)], frame_F0_zero_pad, axis=0)
            split_frame_F0_onset = np.append(
                frame_F0_onset[feature_len_4bars * n_4bars - pad_len: feature_len_4bars *
                               (n_4bars+1)], frame_F0_zero_pad, axis=0)
        else:
            split_waveform = audio[w_start: w_end]
            split_cqt = cqt[feature_len_4bars * n_4bars -
                            pad_len: feature_len_4bars * (n_4bars+1) + pad_len]
            split_stft = stft[feature_len_4bars * n_4bars -
                            pad_len: feature_len_4bars * (n_4bars + 1) + pad_len]
            split_sf = sf[feature_len_4bars * n_4bars -
                            pad_len: feature_len_4bars * (n_4bars + 1) + pad_len]
            split_b = b[feature_len_4bars * n_4bars -
                            pad_len: feature_len_4bars * (n_4bars + 1) + pad_len]
            split_mel_spec = mel_spec[feature_len_4bars * n_4bars -
                                      pad_len: feature_len_4bars * (n_4bars+1) + pad_len]
            split_frame_tab = frame_tab[feature_len_4bars * n_4bars -
                                        pad_len: feature_len_4bars * (n_4bars+1) + pad_len]
            split_frame_tab_onset = frame_tab_onset[feature_len_4bars * n_4bars -
                                                    pad_len: feature_len_4bars * (n_4bars+1) + pad_len]
            split_frame_F0 = frame_F0[feature_len_4bars * n_4bars -
                                      pad_len: feature_len_4bars * (n_4bars+1) + pad_len]
            split_frame_F0_onset = frame_F0_onset[feature_len_4bars * n_4bars -
                                                  pad_len: feature_len_4bars * (n_4bars+1) + pad_len]

        assert split_cqt.shape[0] == split_stft.shape[0] == split_sf.shape[0] == split_b.shape[0] == split_mel_spec.shape[0] == split_frame_F0.shape[
            0] == split_frame_F0_onset.shape[0] == split_frame_tab_onset.shape[0] == split_frame_tab_onset.shape[0]

        np.savez_compressed(split_npz_filename,
                            cqt=split_cqt,
                            stft=split_stft,
                            sf=split_sf,
                            b=split_b,
                            audio=split_waveform,
                            log_cqt=librosa.amplitude_to_db(np.abs(split_cqt)),
                            mel_spec=split_mel_spec,
                            tab=split_tab,
                            tab_onset=split_tab_onset,
                            frame_tab=split_frame_tab,
                            frame_tab_onset=split_frame_tab_onset,
                            F0=split_F0,
                            F0_onset=split_F0_onset,
                            frame_F0=split_frame_F0,
                            frame_F0_onset=split_frame_F0_onset,
                            tempo=tempo,
                            len_in_notes=(note_resolution * 4))
    return