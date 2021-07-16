import os
import yaml
import model as M
from munch import Munch
import numpy as np
import torch
import logging
from argparse import ArgumentParser
from dataset import RawBeatDataset
from utils import setup_seed, y2time, parse_config
from inference import infer

logging.basicConfig(level=logging.INFO)

parser = ArgumentParser()
parser.add_argument("--input_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--beat_type", type=str, default="beat")
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--recover_epoch", type=int, default=None)
parser.add_argument("--seed", type=int, default=2)
parser.add_argument("--filter", action="store_true")
args = parser.parse_args()

setup_seed(args.seed)

train, data_conf, model_conf, optimizer_conf, scheduler_conf = parse_config(args.config, args.beat_type)

def export_result(index_list, beats_list):
    for i, beats in zip(index_list, beats_list):
        with open(os.path.join(args.output_path, "%05d.beat" % (i + 1)), "w") as fout:
            fout.writelines([str(t) + "\n" for t in beats])

if __name__ == "__main__":

    model_name = model_conf.tagging

    beat_dataset =RawBeatDataset(path=args.input_path, hop_length=data_conf.hop_length)
    
    with torch.no_grad():
        if model_conf.recover_epoch is None and args.recover_epoch is None:
            for ckpt in range(5, 21, 5):
                model = M.__getattribute__(model_conf.name)(data_conf.n_frames, data_conf.n_features, data_conf.n_channels, (data_conf.n_grids, data_conf.out_features))
                model = model.cuda()
                model.load_state_dict(torch.load(f"models/{model_name}/model.{ckpt}.pth"))
                index_list, beats_list, _ = infer(beat_dataset, model, args.config, args.beat_type)
                export_result(index_list, beats_list)
        else:
            model = M.__getattribute__(model_conf.name)(data_conf.n_frames, data_conf.n_features, data_conf.n_channels, (data_conf.n_grids, data_conf.out_features))
            model = model.cuda()
            recover_epoch = args.recover_epoch if args.recover_epoch is not None else model_conf.recover_epoch
            load_path = f"{args.beat_type}_models/{model_name}/model.{recover_epoch}.pth"
            model.load_state_dict(torch.load(load_path))
            index_list, beats_list, _ = infer(beat_dataset, model, args.config, args.beat_type)
            export_result(index_list, beats_list)
