#!/usr/bin/env python3
import argparse
import sys

import yaml
from lightning.pytorch import Trainer

from datasets.inference import get_inference_dl_ds
from pl_models import TestPipeline


# from lightning.pytorch.strategies import DDPStrategy


def load_config(config_path):
    with open(config_path, 'r') as input_file:
        config = yaml.safe_load(input_file)

    return config


def parse_args(args):
    parser = argparse.ArgumentParser(description='Template to inference networks with pytorch lightning.')
    parser.add_argument('config', help='path to yaml config file', default='configs/inference.yaml')
    return parser.parse_args(args)


def inference(args=None):
    if args is None:
        args = sys.argv[1:]
        args = parse_args(args)
        config = load_config(args.config)
    else:
        config = args

    dataloader, dataset = get_inference_dl_ds(
        config,
    )

    model = TestPipeline(
        config
    )

    tester = Trainer(
        # strategy=DDPStrategy(find_unused_parameters=False),
        logger=False, **config['trainer']
    )
    tester.test(model, dataloaders=[dataloader])


if __name__ == "__main__":
    inference()
