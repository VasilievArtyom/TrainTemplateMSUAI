import os

import torch
from torch.utils.data import DataLoader

from datasets.train import TrainDataset


class TestDataset(TrainDataset):
    def __init__(self, config, mode='test'):
        config['val_images'] = config['images']
        config['val_annotations'] = config['annotations']
        super().__init__(config, mode='test')
        self.mode = mode
        self.config = config
        self.annotations = []
        self.img_dir = os.path.join(config["data_path"], self.config['images'])
        csv_path = os.path.join(config["annotation_path"], self.config['annotations'])
        self._read_csv(csv_path)
        self._init_transforms()


def collate_fn(batch):
    inputs = torch.stack([b['image'] for b in batch], dim=0)
    ids = torch.stack([b['idx'] for b in batch], dim=0)
    return inputs, ids


def get_inference_dl_ds(config):
    dataset = TestDataset(config)

    dataloader = DataLoader(
        dataset, collate_fn=collate_fn,
        **config['dataloader']
    )
    return dataloader, dataset
