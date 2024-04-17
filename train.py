#!/usr/bin/env python3
import argparse
import os
import shutil
import sys

import yaml
from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
# from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities import rank_zero_only

from datasets.train import get_train_dl_ds
from pl_models import TrainPipeline


def load_config(config_path):
    with open(config_path, 'r') as input_file:
        config = yaml.safe_load(input_file)

    return config


@rank_zero_only
def check_dir(dirname):
    if not os.path.exists(dirname):
        return

    print(f"Save directory - {dirname} exists")
    print("Ignore: Yes[y], No[n]")
    ans = input().lower()
    if ans == 'y':
        shutil.rmtree(dirname)
        return

    raise ValueError("Tried to log experiment into existing directory")


def parse_args(args):
    parser = argparse.ArgumentParser(description='Template for training networks with pytorch lightning.')

    parser.add_argument('config', help='path to yaml config file', default='configs/train.yaml')
    return parser.parse_args(args)


def train(args=None):
    seed_everything(42)
    if args is None:
        args = sys.argv[1:]
        args = parse_args(args)
        config = load_config(args.config)
    else:
        config=args

    config['save_path'] = os.path.join(
        config['exp_path'],
        config['project'],
        config['exp_name']
    )

    check_dir(config['save_path'])
    os.makedirs(config['save_path'], exist_ok=True)

    tensorboard_logger = TensorBoardLogger(
        config['save_path'],
        name='metrics'
    )

    train_loader, _ = get_train_dl_ds(
        config,
        mode='train'
    )

    val_loader, _ = get_train_dl_ds(
        config,
        mode="val"
    )

    model = TrainPipeline(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=config['save_path'],
        save_last=True,
        every_n_epochs=1,
        save_top_k=1,
        save_weights_only=True,
        save_on_train_epoch_end=False,
        **config['checkpoint']
    )

    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        checkpoint_callback
    ]

    trainer = Trainer(
        # strategy=DDPStrategy(find_unused_parameters=False),
        callbacks=callbacks,
        logger=tensorboard_logger,
        **config['trainer']
    )
    trainer.fit(model)


if __name__ == "__main__":
    train()
