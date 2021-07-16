import os
from pytorchtrainer.callback import Callback
from pytorchtrainer.trainer import ModuleTrainer
import torch


class LossAccumulationCallback(Callback):

    def __init__(self, n_steps=100):
        super().__init__(state_attribute_name="train loss", state_attribute_default_value=0)
        self.n_steps = n_steps
        self.total_loss = 0

    def __call__(self, trainer: ModuleTrainer):
        loss = trainer.state.last_train_loss
        self.total_loss += loss
        if trainer.state.current_epoch % self.n_steps == 0:
            avg_loss = self.total_loss / self.n_steps
            setattr(trainer.state, self.state_attribute_name, avg_loss)
            self.total_loss = 0


class SaveModelCallback(Callback):

    def __init__(self, save_path):
        super().__init__(state_attribute_name="saved model", state_attribute_default_value="")
        self.save_path = save_path
    
    def __call__(self, trainer: ModuleTrainer):
        model = trainer.model
        to_save = model.module if hasattr(model, "module") else model
        save_name = f"model.{trainer.state.current_epoch}.pth"
        torch.save(to_save.state_dict(), os.path.join(self.save_path, save_name))
        setattr(trainer.state, self.state_attribute_name, save_name)


class LRSchedulerCallback(Callback):

    def __init__(self, scheduler):
        super().__init__(state_attribute_name="lr", state_attribute_default_value=0)
        self.scheduler = scheduler
    
    def __call__(self, trainer: ModuleTrainer):
        self.scheduler.step()
        setattr(trainer.state, self.state_attribute_name, self.scheduler.get_lr())
