"""
Configuration Utils Function Collection
"""
import os
import argparse
import collections
from datetime import datetime

import yaml
from dotmap import DotMap


def get_config(config_name=None):
    """
    Reads the configuration and returns a configuration DotMap
    :param config_name: Optional name of the configuration file
    :return: Configuration DotMap with arguments parsed form terminal/cmd
    """

    # Read arguments
    args = _read_arguments()

    # Read the config name specified
    if not config_name:
        config_name = args.__dict__['config.name']

    # load the file and parse it to a DotMap
    with open("config/" + config_name + ".yml", "r") as file:
        config_dict = yaml.safe_load(file)
    config = DotMap(config_dict)

    # Overwrite default values
    return _overwrite_defaults(config, args)


def _overwrite_defaults(config, args):
    """
    Overwrite the default values in the configuration DotMap
    :param config: configuration file
    :param args: command line arguments
    :return: DotMap Configuration with new values
    """

    # Overwrite all arguments that are set via terminal/cmd
    for argument, argument_value in args.__dict__.items():
        if argument_value is not None:
            config = _replace_value_in_config(config, argument, argument_value)
    return config


def _read_arguments():
    """
    Read the arguments from the command line/terminal

    :return: ArgParser
    """
    parser = argparse.ArgumentParser(description='Arguments for MGS Detector')
    parser.add_argument('--config.name', default='train/efficient_net', type=str)
    parser.add_argument('--basic.cuda_device_name', default=None, type=str)
    parser.add_argument('--basic.experiment_name', default=None, type=str)
    parser.add_argument(
        '--basic.experiment_description',
        default=None,
        type=str)
    parser.add_argument('--basic.result_directory', default=None, type=str)
    parser.add_argument(
        '--basic.enable_wand_reporting',
        default=None,
        type=int)

    parser.add_argument('--model.name', default=None, type=str)
    parser.add_argument('--model.pretrained', default=None, type=int)
    parser.add_argument('--model.timm_model_name', default=None, type=int)
    parser.add_argument('--model.dropout', default=None, type=float)
    parser.add_argument('--model.mgs_attributes', default=None, type=int)
    parser.add_argument('--model.mgs_one_hot', default=None, type=int)
    parser.add_argument('--model.freeze_layers', default=None, type=int)

    parser.add_argument(
        '--dataset.partition_filename',
        default=None,
        type=str)
    parser.add_argument(
        '--dataset.dataset_labels_filename',
        default=None,
        type=str)
    parser.add_argument(
        '--dataset.dataset_image_folder',
        default=None,
        type=str)

    parser.add_argument('--dataset.bounding_box_scale', default=None, type=int)
    parser.add_argument('--training.epochs', default=None, type=int)
    parser.add_argument('--training.save_frequency', default=None, type=int)
    parser.add_argument('--training.optimizer.type', default=None, type=str)
    parser.add_argument(
        '--training.optimizer.learning_rate',
        default=None,
        type=float)
    parser.add_argument(
        '--training.optimizer.momentum',
        default=None,
        type=float)
    parser.add_argument(
        '--training.optimizer.beta1',
        default=None,
        type=float)
    parser.add_argument(
        '--training.optimizer.beta3',
        default=None,
        type=float)
    parser.add_argument(
        '--training.optimizer.epsilon',
        default=None,
        type=float)
    parser.add_argument(
        '--training.optimizer.weight_decay',
        default=None,
        type=float)
    parser.add_argument('--training.criterion.type', default=None, type=str)
    parser.add_argument('--training.lr_scheduler.type', default=None, type=str)
    parser.add_argument(
        '--training.lr_scheduler.step_size',
        default=None,
        type=int)
    parser.add_argument(
        '--training.lr_scheduler.gamma',
        default=None,
        type=float)
    parser.add_argument(
        '--training.lr_scheduler.patience',
        default=None,
        type=float)

    parser.add_argument(
        '--training.dataloader.batch_size',
        default=None,
        type=int)
    parser.add_argument(
        '--training.dataloader.shuffle',
        default=None,
        type=str)
    parser.add_argument(
        '--training.dataloader.num_workers',
        default=None,
        type=int)
    parser.add_argument(
        '--training.dataloader.prefetch_factor',
        default=None,
        type=int)

    parser.add_argument(
        '--training.save_preprocessed_image.enabled',
        default=None,
        type=int)
    parser.add_argument(
        '--training.save_preprocessed_image.frequency',
        default=None,
        type=int)


    args = parser.parse_args()
    return args


def _replace_value_in_config(config, argument, argument_value):
    """
    Replaces a value in the DotMap
    :param config: Configuration DotMap
    :param argument: Argument to overwrite
    :param argument_value: Argument value
    :return: new DotMap with new Values
    """

    # Recursive Help function which creates a nested dict
    def _create_nested_dict(key, value):
        value = {key[-1]: value}
        new_key_list = key[0:-1]
        if len(new_key_list) > 1:
            return _create_nested_dict(new_key_list, value)
        return {new_key_list[0]: value}

    # Recursive Help function which updates a value
    def _update(key, value):
        for k, val in value.items():
            if isinstance(val, collections.abc.Mapping):
                key[k] = _update(key.get(k, {}), val)
            else:
                key[k] = val
        return key

    argument_keys = argument.split('.')
    new_dict = _create_nested_dict(argument_keys, argument_value)
    return DotMap(_update(config.toDict(), new_dict))


def create_result_directory(config):
    """
    Creates the result directory and updates the configuration file
    :param config: configuration file
    """

    # Get current time
    now = datetime.now()

    # Create name of the directory using a the current time as well as the
    # experiment name
    directory_name = '{}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}-{}'.format(
        now.year,
        now.month,
        now.day,
        now.hour,
        now.minute,
        now.second,
        config.basic.experiment_name)

    # Join directory path
    result_directory = os.path.join(
        config.basic.result_directory, directory_name)
    if not os.path.exists(result_directory):
        try:
            os.makedirs(result_directory)
            config.basic.result_directory = result_directory
            config.basic.result_directory_name = directory_name
            save_config_to_file(config)
        except BaseException as exc:
            raise Exception('directory could not be created') from exc
    else:
        raise Exception('directory already exists')


def save_config_to_file(config):
    """
    Save configuration to disk
    :param config: DotMap Configuration
    """
    file = open(os.path.join(config.basic.result_directory, 'config.yml'), "w")
    file.write(yaml.dump(config.toDict()))
    file.close()
