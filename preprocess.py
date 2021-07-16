import os
import torch
import numpy as np
from argparse import ArgumentParser
import librosa
import librosa.display
import matplotlib.pyplot as plt

def norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def load_beat(beat_file):
    return np.array([float(line.strip()) for line in open(beat_file, "r").readlines()])

def get_spectrum(audio_file, hop_length):
    data, fs = librosa.load(audio_file)
    data = librosa.feature.melspectrogram(data, fs, hop_length=hop_length)
    data = librosa.power_to_db(data, ref=np.max)
    data = data.T
    data = norm(data)
    return torch.from_numpy(data).to(torch.float32)

def get_frame_beat(beat_file, hop_length, fs):
    beat = load_beat(beat_file)
    beat = beat * fs / hop_length
    beat = beat.astype(np.int)
    return torch.from_numpy(beat).to(torch.float32)

def get_norm_beat(beat_file):
    beat = load_beat(beat_file)
    beat = norm(beat)
    return torch.from_numpy(beat).to(torch.float32)
