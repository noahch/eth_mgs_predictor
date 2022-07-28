#!/usr/bin/python3
"""
Main File to start training the AFFACT Network
"""
import torch.utils.data

from preprocessing.dataset_generator import _get_partition_ids
from preprocessing.mgs_dataset import MGSDataset
from utils.config_utils import get_config, create_result_directory
from training.train_model import TrainModel
from utils.utils import init_environment
import time
import os
import pandas as pd
from PIL import Image
import torchvision.transforms as T
def main():
    """
    Run training for a specific model
    """
    # Load configuration for training
    config = get_config()
    # Init environment, use GPU if available, set random seed
    device = init_environment(config)
    # Create result directory
    create_result_directory(config)
    # Create a training instance with the loaded configuration on the loaded device
    training_instance = TrainModel(config, device)
    # get model
    m = training_instance.get_model()

    # use .module to get rid of Data Parallel
    m = m.module.cpu()
    m.eval()

    val_transforms = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # for i in range(10,99):
    #
    #     img_name = f'C:\\Users\\Hackerman\\Documents\\Noah\\BMv1\\images\\0000{i}.jpg'
    #     image = Image.open(img_name).convert('RGB')
    #
    #     image = val_transforms(image)
    #     image = image.unsqueeze(0)
    #     since = time.time()
    #     out = m(image)
    #     print(time.time() - since)

    partition_df = pd.read_csv("C:\\Users\\Hackerman\\Documents\\Noah\\BMv1\\partitions.csv", index_col=0)
    dataset = MGSDataset(ids=_get_partition_ids(partition_df, 'test'), config=config, transform=val_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, persistent_workers=True, **config.training.dataloader)

    for inputs, labels in dataloader:
        since = time.time()
        out = m(inputs)
        print(time.time() - since)


if __name__ == '__main__':
    main()