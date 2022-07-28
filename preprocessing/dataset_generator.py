"""
functions that create datasets for training, validation and testing
"""
import os
import random
import math
import pandas as pd
import torch
import torch.utils.data
from timm.data import RandomResizedCropAndInterpolation
from torchvision.transforms import transforms, Normalize, RandomHorizontalFlip, ColorJitter
# from preprocessing.image_transformer import ImageTransformer
from preprocessing.mgs_dataset import MGSDataset

import torchvision.transforms as T

def get_train_val_dataset(config):
    """
    generate dataloader, dataset and all meta information needed for training
    :param config: the configuration file
    :return: result dict containing dataloader, dataset size, attribute baseline accuracy, dataset meta information
    """
    # Loads the labels and partition df from disk
    labels, partition_df = _load_dataframes(config)

    # Gets the training ids from the partition file
    train_ids = _get_partition_ids(partition_df, 'train')

    # Gets the training ids from the partition file
    val_ids = _get_partition_ids(partition_df, 'val')

    # Define the transformations that are applied to each image
    train_transforms = T.Compose([
        T.RandomRotation(degrees=4, interpolation=T.InterpolationMode.BILINEAR),
        # T.Resize((256, 256), interpolation=T.InterpolationMode.BILINEAR),
        RandomResizedCropAndInterpolation(size=(224, 224), scale=(0.9, 1.1), ratio=(0.75, 1.3333),
                                          interpolation='bilinear'),
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=[0.8, 1.2], contrast=[0.8, 1.2], saturation=[0.8, 1.2]),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Define the transformations that are applied to each image
    val_transforms = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Generates the data for training
    dataset_train, training_loader = generate_dataset_and_loader(train_transforms, train_ids, config)

    # Generates the data for validation
    dataset_val, validation_loader = generate_dataset_and_loader(val_transforms, val_ids, config)

    # df = pd.read_csv(os.path.join('C:\\Users', 'Hackerman', 'Documents', 'Noah', 'BMv1', 'labels.csv'))
    # unique_ids = df['id'].unique().tolist()
    # result = pd.DataFrame(columns=['run', 'training_ids', 'test_ids', 'acc', 'tpr', 'tnr'])
    # random.seed(10)
    # random.shuffle(unique_ids)
    # transform = T.Compose([
    #     # T.Resize((224,224), interpolation=T.InterpolationMode.BILINEAR),
    #     T.ToTensor(),
    #     T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    # )
    #
    # ds_train = MousePainDataset(csv_file=os.path.join('C:\\Users', 'Hackerman', 'Documents', 'Noah', 'BMv1', 'labels.csv'),
    #                       root_dir=os.path.join('C:\\Users', 'Hackerman', 'Documents', 'Noah', 'BMv1', 'preprocessed_bw'),
    #                       ids=unique_ids[:math.floor(len(unique_ids)*0.8)],
    #                       transform=transform)
    #
    # train_loader = torch.utils.data.DataLoader(ds_train, batch_size=100, shuffle=True, num_workers=8, pin_memory=True)
    # ds_val = MousePainDataset(csv_file=os.path.join('C:\\Users', 'Hackerman', 'Documents', 'Noah', 'BMv1', 'labels.csv'),
    #                       root_dir=os.path.join('C:\\Users', 'Hackerman', 'Documents', 'Noah', 'BMv1', 'preprocessed_bw'),
    #                       ids=unique_ids[math.floor(len(unique_ids)*0.8):],
    #                       transform=transform)
    #
    # val_loader = torch.utils.data.DataLoader(ds_val, batch_size=100, shuffle=True, num_workers=8, pin_memory=True)
    #

    dataloaders = {
        'train': training_loader,
        'val': validation_loader
    }

    dataset_sizes = {
        'train': len(dataset_train),
        'val': len(dataset_val)
    }

    # dataset_meta_information = {
    #     'label_names': dataset_train.get_label_names(),
    #     'number_of_labels': len(dataset_train.get_label_names())
    # }

    result_dict = dict()
    result_dict['dataloaders'] = dataloaders
    result_dict['dataset_sizes'] = dataset_sizes
    # result_dict['dataset_meta_information'] = dataset_meta_information
    return result_dict

# def generate_test_dataset(config):
#     """
#     Generate dataframe for testing
#     :param config: Configuration file
#     :return: labels, landmarks, bounding boxes of test partition
#     """
#     # Loads the bounding boxes, labels, landmarks and partition df from disk
#     bounding_boxes, labels, landmarks, partition_df = _load_dataframes(config)
#
#     # Gets the training labels, landmarks, bounding boxes according to the partition file
#     df_test_labels, df_test_landmarks, df_test_bounding_boxes = _get_partition_dataframes(partition_df,
#                                                                                              2,
#                                                                                              labels,
#                                                                                              landmarks,
#                                                                                              bounding_boxes)
#     return df_test_labels, df_test_landmarks, df_test_bounding_boxes

def _load_dataframes(config):
    """
    loads the labels and partition dataframe from disk
    :param config: the configuration file
    :return: labels, partition dataframe
    """
    labels = pd.read_csv(config.dataset.dataset_labels_filename)
    partition_df = pd.read_csv(config.dataset.partition_filename, index_col=0)

    return labels, partition_df


def _get_partition_ids(partition_df, partition):
    """
    get unique mouse id based on partition (train/val/test)
    :param partition_df: partition dataframe
    :param partition: train/val/test
    """

    # Filter on partition
    filtered_partition_df = partition_df.loc[partition_df["partition"] == partition]

    return filtered_partition_df['id'].unique().tolist()


def generate_dataset_and_loader(transform, ids, config):
    """
    creates a dataset instance of the MGSDataset given the transformer, the unique mouse ids and the configuration file
    :param transform: Transformations to be applied to the images
    :param ids: unique ids of the mice
    :param config: the configuration file
    :return: the dataset and dataloader
    """
    dataset = MGSDataset(ids=ids, config=config, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, persistent_workers=True, **config.training.dataloader)
    return dataset, dataloader

# Dataset -> # full_mgs, pain_only, label file, image_base_path
