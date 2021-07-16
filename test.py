import os
import model as M
import numpy as np
import torch
import logging
from argparse import ArgumentParser
from dataset import FastSlowRawBeatDataset, RawBeatDataset
from sklearn.model_selection import train_test_split
import eval.beat_evaluation_toolbox as eval_tool
from utils import setup_seed, parse_config
from inference import infer, fast_slow_infer

logging.basicConfig(level=logging.INFO)

parser = ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--recover_epoch", type=int, default=None)
parser.add_argument("--range", type=str, default="5,21,5")
parser.add_argument("--seed", type=int, default=2)
parser.add_argument("--beat_type", type=str, default="beat")
parser.add_argument("--fast_slow", action="store_true")
args = parser.parse_args()

setup_seed(args.seed)

train, data_conf, model_conf, optimizer_conf, scheduler_conf = parse_config(args.config, args.beat_type)

def export_metrics(anns_list, beats_list):
    res = eval_tool.evaluate_db(anns_list, beats_list)
    res = { k : np.mean(res["scores"][k]) for k in res["scores"] }
    metrics = [res["fMeasure"], res["cemgilAcc"], res["pScore"], res["cmlC"], res["cmlT"], res["amlC"], res["amlT"]]
    line = "\t".join([str(m) for m in metrics]) + "\n"
    with open(f"metrics/{model_name}.{args.beat_type}.txt", ["w", "a"][os.path.exists(f"metrics/{model_name}.{args.beat_type}.txt")]) as fout:
        fout.write(line)

if __name__ == "__main__":

    model_name = model_conf.tagging

    DS = FastSlowRawBeatDataset if args.fast_slow else RawBeatDataset
    dataset = DS(path="dataset", hop_length=data_conf.hop_length)
    dataset_train, dataset_test = train_test_split(dataset, test_size=0.1, random_state=12)

    infer_func = fast_slow_infer if args.fast_slow else infer

    with torch.no_grad():
        if args.recover_epoch is None:
            if args.range is None:
                exit(0)
            for ckpt in range(*[int(i) for i in args.range.split(",")]):
                model = M.__getattribute__(model_conf.name)(data_conf.n_frames, data_conf.n_features, data_conf.n_channels, (data_conf.n_grids, data_conf.out_features))
                model = model.cuda()
                load_path = f"{args.beat_type}_models/{model_name}/model.{ckpt}.pth"
                model.load_state_dict(torch.load(load_path))
                _, beats_list, anns_list = infer_func(dataset_test, model, args.config, args.beat_type)
                export_metrics(anns_list, beats_list)
        else:
            model = M.__getattribute__(model_conf.name)(data_conf.n_frames, data_conf.n_features, data_conf.n_channels, (data_conf.n_grids, data_conf.out_features))
            model = model.cuda()
            recover_epoch = args.recover_epoch if args.recover_epoch is not None else model_conf.recover_epoch
            load_path = f"{args.beat_type}_models/{model_name}/model.{recover_epoch}.pth"
            model.load_state_dict(torch.load(load_path))
            _, beats_list, anns_list = infer_func(dataset_test, model, args.config, args.beat_type)
            export_metrics(anns_list, beats_list)
