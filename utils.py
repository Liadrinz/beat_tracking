import numpy as np
import torch
import random
import librosa
import yaml
from munch import Munch

_, fs = librosa.load("dataset/wav/00001.wav")

def parse_config(config_file, beat_type="beat"):
    config = yaml.load(open(config_file))
    train = Munch(config["train"])
    data_conf = Munch(train.data)
    model_conf = Munch(train.model)
    optimizer_conf = Munch(train.optimizer)
    scheduler_conf = Munch(train.scheduler)
    if beat_type == "downbeat":
        data_conf.n_grids //= 4
    return train, data_conf, model_conf, optimizer_conf, scheduler_conf

def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def y2time(y, hop_length, n_frames, n_grids, threshold=0.5):
    grids_per_unit = n_frames // n_grids
    beat_indices = torch.where(y[:, 1] >= threshold)[0]  # which grids beats are in
    beat_positions = beat_indices.to(torch.float32) + y[:, 0][beat_indices]  # plus offset ratio (0, 1)
    beat_positions = beat_positions * grids_per_unit  # scale up to feature frame level
    beat_positions = beat_positions * hop_length  # scale up to real frame level
    beat_times = beat_positions / fs
    return beat_times

def batch_to_tensor(batch, device=None, non_blocking=False, dtype=None):
    X, Y = batch
    X = [x.to(device=device, dtype=dtype, non_blocking=non_blocking) for x in X]
    Y = [y.to(device=device, dtype=dtype, non_blocking=non_blocking) for y in Y]
    return X, Y, dict()
