from dataset import RawBeatDataset, FastSlowRawBeatDataset
import torch
from utils import parse_config, y2time

def get_beats(model, X, data_conf, beat_type):
    cat_frames = data_conf.n_frames - X.shape[0] % data_conf.n_frames
    X_cat = torch.zeros((cat_frames, X.shape[1]))
    X = torch.cat((X, X_cat))
    X = X.cuda()
    outputs = model(X.view(-1, data_conf.n_frames, X.shape[-1]))
    outputs = outputs.view(-1, data_conf.out_features)
    outputs = outputs[:len(outputs)-cat_frames//(data_conf.n_frames//data_conf.n_grids), :]
    if beat_type == "downbeat": threshold = 0.01
    else: threshold = 0.5
    beats = y2time(outputs, data_conf.hop_length, data_conf.n_frames, data_conf.n_grids, threshold=threshold)
    return beats

def infer(dataset: RawBeatDataset, model, config_file, beat_type):
    data_conf = parse_config(config_file, beat_type)[1]
    index_list = []
    beats_list = []
    anns_list = []
    for X, yb, ydb, index in dataset:
        beats = get_beats(model, X, data_conf, beat_type)
        beats = beats.cpu().numpy()
        if beat_type == "beat": anns = yb
        elif beat_type == "downbeat": anns = ydb
        else: raise RuntimeError("Unexpected beat type")
        index_list.append(index)
        beats_list.append(beats)
        anns_list.append(anns)
    return index_list, beats_list, anns_list

def fast_slow_get_beats(model, X_fast, X_slow, data_conf):
    fast_cat_frames = data_conf.n_frames - X_fast.shape[0] % data_conf.n_frames
    X_cat = torch.zeros((fast_cat_frames, X_fast.shape[1]))
    X_fast = torch.cat((X_fast, X_cat))
    X_fast = X_fast.cuda()
    X_fast = X_fast.view(-1, data_conf.n_frames, X_fast.shape[-1])

    slow_cat_frames = (data_conf.n_frames * 2) - X_slow.shape[0] % (data_conf.n_frames * 2)
    X_cat = torch.zeros((slow_cat_frames, X_slow.shape[1]))
    X_slow = torch.cat((X_slow, X_cat))
    X_slow = X_slow.cuda()
    X_slow = X_slow.view(-1, data_conf.n_frames * 2, X_slow.shape[-1])

    outputs = model((X_fast, X_slow))
    outputs = outputs.view(-1, data_conf.out_features)
    outputs = outputs[:len(outputs)-fast_cat_frames//(data_conf.n_frames//data_conf.n_grids), :]
    beats = y2time(outputs, data_conf.hop_length, data_conf.n_frames, data_conf.n_grids)
    return beats

def fast_slow_infer(dataset: FastSlowRawBeatDataset, model, config_file, beat_type):
    data_conf = parse_config(config_file, beat_type)[1]
    index_list = []
    beats_list = []
    anns_list = []
    for X, yb, ydb, index in dataset:
        X_fast, X_slow = X
        yb, ydb = yb[0], ydb[0]
        beats = fast_slow_get_beats(model, X_fast, X_slow, data_conf)
        beats = beats.cpu().numpy()
        if beat_type == "beat": anns = yb
        elif beat_type == "downbeat": anns = ydb
        else: raise RuntimeError("Unexpected beat type")
        index_list.append(index)
        beats_list.append(beats)
        anns_list.append(anns)
    return index_list, beats_list, anns_list
