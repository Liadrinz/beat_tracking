import os

import torch
import model as M
import dataset as D
import loss as L
import logging
from torch import optim
from torch.utils.data.dataloader import DataLoader
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from utils import setup_seed, parse_config
import pytorchtrainer as ptt
import pytorchtrainer.utils as ptt_utils
import utils
import callback

logging.basicConfig(level=logging.INFO)

parser = ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--seed", type=int, default=2)
parser.add_argument("--beat_type", type=str, default="beat")
args = parser.parse_args()

setup_seed(args.seed)

train, data_conf, model_conf, optimizer_conf, scheduler_conf = parse_config(args.config, args.beat_type)

device = train.device if hasattr(train, "device") else 0

os.makedirs(os.path.join("beat_models", model_conf.tagging), exist_ok=True)
os.makedirs(os.path.join("downbeat_models", model_conf.tagging), exist_ok=True)

if args.beat_type not in ["beat", "downbeat"]:
    raise RuntimeError("Unexpected beat type")

if __name__ == "__main__":
    prepare_batch_function = utils.batch_to_tensor if model_conf.name == "FastSlowResNeXtBeatModel" else ptt_utils.batch_to_tensor

    dataset = D.__getattribute__(data_conf.dataset)(path="dataset", beat_type=args.beat_type, frames=data_conf.n_frames, grids=data_conf.n_grids, hop_length=data_conf.hop_length, fs=data_conf.fs)
    dataset_train, dataset_test = train_test_split(dataset, test_size=0.1, random_state=12)
    train_loader = DataLoader(dataset_train, batch_size=train.batch_size, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=train.batch_size, shuffle=True)

    model = M.__getattribute__(model_conf.name)(data_conf.n_frames, data_conf.n_features, data_conf.n_channels, (data_conf.n_grids, data_conf.out_features))
    criterion = L.__getattribute__(train.loss)()
    optimizer = optim.__getattribute__(optimizer_conf.name)(model.parameters(), **optimizer_conf.params)
    scheduler = optim.lr_scheduler.__getattribute__(scheduler_conf.name)(optimizer, **scheduler_conf.params)

    trainer = ptt.create_default_trainer(model, optimizer, criterion, device, prepare_batch_function=prepare_batch_function)
    trainer.register_post_iteration_callback(callback.LossAccumulationCallback())
    trainer.register_post_epoch_callback(ptt.callback.ValidationCallback(test_loader, ptt.metric.TorchLoss(criterion)))
    trainer.register_post_epoch_callback(callback.SaveModelCallback(os.path.join(f"{args.beat_type}_models", model_conf.tagging)))
    trainer.register_post_epoch_callback(callback.LRSchedulerCallback(scheduler))
    if model_conf.recover_epoch is not None:
        state_dict = torch.load(os.path.join(f"{args.beat_type}_models", model_conf.tagging, f"model.{model_conf.recover_epoch}.pth"))
        model.load_state_dict(state_dict)
        trainer.state.current_epoch = model_conf.recover_epoch
        scheduler.step(model_conf.recover_epoch)
    trainer.train(train_loader, max_epochs=train.n_epochs)
