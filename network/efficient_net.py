import timm
from torch import nn
import torch

def get_model(model_name, freeze_layers=False):
    # model = timm.create_model('tf_efficientnet_b0', pretrained=True, num_classes=2)
    model = timm.create_model(model_name, pretrained=True, num_classes=2)
    if freeze_layers:
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
    return model


def get_model_modified(model_name, freeze_layers=False):
    model = CustomEfficientnet(model_name)

    if freeze_layers:
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

    return model


class CustomEfficientnet(nn.Module):
    def __init__(self, model_name):
        super(CustomEfficientnet, self).__init__()
        # self.model = timm.create_model('tf_efficientnet_b0', pretrained=True, num_classes=2)
        self.model = timm.create_model(model_name, pretrained=True, num_classes=2)
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        # TODO: Softmax
        self.fc1 = nn.Sequential(
                        nn.Linear(num_ftrs, 512),
                        nn.Linear(512, 3),
                        nn.Softmax(dim=1),
                    )
        self.fc2 = nn.Sequential(
                        nn.Linear(num_ftrs, 512),
                        nn.Linear(512, 3),
                        nn.Softmax(dim=1),
                    )
        self.fc3 = nn.Sequential(
                        nn.Linear(num_ftrs, 512),
                        nn.Linear(512, 3),
                        nn.Softmax(dim=1),
                    )
        self.fc4 = nn.Sequential(
                        nn.Linear(num_ftrs, 512),
                        nn.Linear(512, 3),
                        nn.Softmax(dim=1),
                    )
        self.fc5 = nn.Sequential(
                        nn.Linear(num_ftrs, 512),
                        nn.Linear(512, 3),
                        nn.Softmax(dim=1),
                    )

    def forward(self, x):
        x = self.model(x)
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        out3 = self.fc3(x)
        out4 = self.fc4(x)
        out5 = self.fc5(x)
        return torch.cat((out1, out2, out3, out4, out5), dim=1)
