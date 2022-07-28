"""
Model Manager Class which helps setting up the model for training
"""
import torch
from torch import nn
from network.efficient_net import get_model, get_model_modified


class ModelManager():
    """
    Model Manager Class
    """

    def __init__(self, config, device):
        """
        Init Model Manager
        :param config: DotMap Configuration
        :param device: cuda device
        """
        self.config = config
        self.device = device

        # Get model for training on multiple GPUs
        self.model = nn.DataParallel(self.__get_model(), device_ids=[int(
            x[-1]) for x in self.config.basic.cuda_device_name.split(',')])

        # if self.config.model.affact_weights and self.config.model.name == 'affact_ext':
        #     state_dict = torch.load(self.config.model.affact_weights, map_location=self.config.basic.cuda_device_name.split(',')[0])
        #     self.model.load_state_dict(state_dict['model_state_dict'], strict=False)

        # Transfer model to GPU
        self.model_device = self.model.to(self.device)

    def __get_model(self):
        """
        Get the model based on configuration Value

        :return: A model
        """

        if self.config.model.name == "efficient_net":
            return get_model(self.config.model.timm_model_name, self.config.model.freeze_layers)
        if self.config.model.name == "efficient_net_mod":
            return get_model_modified(self.config.model.timm_model_name, self.config.model.freeze_layers)


        raise Exception("Model {} does not exist".format(self.config.model.name))

    def get_model(self):
        return self.model


