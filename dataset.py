import os
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from preprocess import get_spectrum, get_frame_beat, get_norm_beat
from utils import flip


class PointBeatDataset(Dataset):

    def __init__(self, path="dataset", beat_type=None, length=100, frames=256, grids=8, hop_length=512, fs=22050):
        self.beat_type = beat_type
        self.length = length
        self.frames = frames
        self.grids = grids
        self.wav_list = [get_spectrum(os.path.join(path, "wav/%05d.wav" % (i + 1)), hop_length) for i in range(length)]
        self.beat_list = [get_norm_beat(os.path.join(path, "beat", "%05d.beat" % (i + 1)), hop_length, fs) for i in range(length)]
        self.downbeat_list = [get_norm_beat(os.path.join(path, "downbeat", "%05d.downbeat" % (i + 1)), hop_length, fs) for i in range(length)]
    
    def __getitem__(self, index: int):
        index %= 100
        if self.beat_type == "beat":
            return self.wav_list[index], self.beat_list[index]
        if self.beat_type == "downbeat":
            return self.wav_list[index], self.downbeat_list[index]
        raise RuntimeError("Unexpected beat type")
        

class FastSlowBeatDataset(Dataset):

    def __init__(self, path="dataset", beat_type=None, length=100, frames=256, grids=8, hop_length=512, fs=22050):
        self.beat_type = beat_type
        self.fast_dataset = BeatDataset(path=path, beat_type=beat_type, length=length, frames=frames, grids=grids, hop_length=hop_length, fs=fs)
        self.slow_dataset = BeatDataset(path=path, beat_type=beat_type, length=length, frames=frames*2, grids=grids, hop_length=hop_length//2, fs=fs)
    
    def __getitem__(self, index: int):
        if self.beat_type is not None:
            X_fast, y_fast = self.fast_dataset[index]
            X_slow, y_slow = self.slow_dataset[index]
            return (X_fast, X_slow), y_slow
        raise RuntimeError("Unexpected beat type")

    def __len__(self):
        return len(self.fast_dataset)

class BeatDataset(Dataset):

    def __init__(self, path="dataset", beat_type=None, length=100, frames=256, grids=8, hop_length=512, fs=22050):
        self.beat_type = beat_type
        self.length = length
        self.frames = frames
        self.grids = grids
        self.wav_list = [get_spectrum(os.path.join(path, "wav/%05d.wav" % (i + 1)), hop_length) for i in range(length)]
        self.beat_list = [get_frame_beat(os.path.join(path, "beat", "%05d.beat" % (i + 1)), hop_length, fs) for i in range(length)]
        self.downbeat_list = [get_frame_beat(os.path.join(path, "downbeat", "%05d.downbeat" % (i + 1)), hop_length, fs) for i in range(length)]

    def _get_y_beat(self, beat_type, index, clip_begin, clip_end, grids_per_unit):
        if beat_type == "beat":
            beat = self.beat_list[index]
        elif beat_type == "downbeat":
            beat = self.downbeat_list[index]
        beat = beat.to(torch.long)
        valid_beat_indices = np.where((beat >= clip_begin) * (beat < clip_end))[0]
        beat = beat[valid_beat_indices] - clip_begin
        beat_grid_indices = beat // grids_per_unit
        beat_grid_offsets = beat % grids_per_unit / grids_per_unit
        y_beat = torch.zeros((self.grids, 2))
        y_beat[beat_grid_indices, 0] = beat_grid_offsets
        y_beat[beat_grid_indices, 1] = 1
        return y_beat

    def __getitem__(self, index):
        offset = index // 100
        index %= 100
        wav = self.wav_list[index]
        clip_begin = np.clip(offset, 0, len(wav) - self.frames)
        clip_end = clip_begin + self.frames
        wav_clip = wav[clip_begin:clip_end]
        grids_per_unit = self.frames // self.grids
        y_beat = self._get_y_beat("beat", index, clip_begin, clip_end, grids_per_unit)
        y_downbeat = self._get_y_beat("downbeat", index, clip_begin, clip_end, grids_per_unit)
        if self.beat_type == "beat":
            return wav_clip, y_beat
        elif self.beat_type == "downbeat":
            return wav_clip, y_downbeat
        return wav_clip, y_beat, y_downbeat, offset, index
    
    def __len__(self):
        return self.length * 1200


class AugBeatDataset(BeatDataset):

    def __getitem__(self, index):
        data = super().__getitem__(index // 2)
        if index % 2 == 0:
            return data
        if len(data) == 5:
            wav_clip, y_beat, y_downbeat, offset, index = data
            return flip(wav_clip, dim=0), flip(y_beat, dim=0), flip(y_downbeat, dim=0), offset, index
        wav_clip, y = data
        wav_clip = flip(wav_clip, dim=0)
        y = flip(y, dim=0)
        return wav_clip, y
    
    def __len__(self):
        return super().__len__() * 2


class RawBeatDataset(Dataset):

    def __init__(self, path="dataset", length=100, hop_length=51):
        self.length = length
        self.wav_list = [get_spectrum(os.path.join(path, "wav/%05d.wav" % (i + 1)), hop_length) for i in range(length)]
        self.beat_list = [np.array([float(line.strip()) for line in open(os.path.join(path, "beat", "%05d.beat" % (i + 1)), "r").readlines()]) for i in range(length)]
        self.downbeat_list = [np.array([float(line.strip()) for line in open(os.path.join(path, "downbeat", "%05d.downbeat" % (i + 1)), "r").readlines()]) for i in range(length)]
    
    def __getitem__(self, index):
        wav, beat, downbeat = self.wav_list[index], self.beat_list[index], self.downbeat_list[index]
        return wav, beat, downbeat, index
    
    def __len__(self):
        return self.length


class FastSlowRawBeatDataset(Dataset):

    def __init__(self, path="dataset", length=100, hop_length=512):
        self.fast_dataset = RawBeatDataset(path=path, length=length, hop_length=hop_length)
        self.slow_dataset = RawBeatDataset(path=path, length=length, hop_length=hop_length//2)
    
    def __getitem__(self, index: int):
        return tuple(zip(self.fast_dataset[index], self.slow_dataset[index]))
    
    def __len__(self):
        return len(self.fast_dataset)
