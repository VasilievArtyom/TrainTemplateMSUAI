import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms_by_config(aug_config):
    augs = []
    for item in aug_config:
        augs.append(getattr(A, item['name'])(**item['kwargs']))
    return A.Compose(augs)


def transforms(transforms_config):
    return get_transforms_by_config(transforms_config)


def preprocess(preprocess_config):
    return A.Compose([
        A.PadIfNeeded(
            min_height=preprocess_config['side_size'],
            min_width=preprocess_config['side_size'],
            value=0
        ),
        A.Normalize(
            mean=preprocess_config['mean'],
            std=preprocess_config['std']
        ),
        ToTensorV2()
    ])
