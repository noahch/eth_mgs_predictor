"""
Class that handles training of the model
"""
import copy
import os
import time

import torch
from torch import optim, nn
from torch.optim import lr_scheduler

from preprocessing.dataset_generator import get_train_val_dataset
from training.model_manager import ModelManager
import wandb
import logging
from tqdm import tqdm

from sklearn.metrics import confusion_matrix
import numpy as np

class TrainModel(ModelManager):
    """
    Class that manages the training of models
    """

    def __init__(self, config, device):
        """
        init
        :param config: the training configuration file
        :param device: the device
        """
        super().__init__(config, device)

        # get the training and validation datasets
        self.datasets = get_train_val_dataset(config)
        # get the optimizer
        self.optimizer = self._get_optimizer()
        # get the loss criterion
        self.criterion = self._get_criterion()
        # get the learning rate scheduler
        self.lr_scheduler = self._get_lr_scheduler()

    def train(self):
        """
        train the model
        :return: result of a specific training process (trained model)
        """

        if self.config.model.name == "efficient_net":
            return self._train3()

        elif self.config.model.name == "efficient_net_mod":
            return self._train3()


        else:
            raise Exception("Model {} does not have a training routine".format(self.config.model.name))

    def _get_lr_scheduler(self):
        """
        get the learning rate scheduler
        :return: LR Scheduler
        """

        # Step learning rate
        if self.config.training.lr_scheduler.type == "StepLR":
            return lr_scheduler.StepLR(self.optimizer,
                                       step_size=self.config.training.lr_scheduler.step_size,
                                       gamma=self.config.training.lr_scheduler.gamma)

        # Reduce on plateau learning rate
        elif self.config.training.lr_scheduler.type == "ReduceLROnPlateau":
            return lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                  patience=self.config.training.lr_scheduler.patience,
                                                  factor=self.config.training.lr_scheduler.gamma, cooldown=1)
        raise Exception("Scheduler {} does not exist".format(self.config.training.lr_scheduler.type))

    def _save_model(self, model_state_dict, optimizer_state_dict, filename):
        """
        Saves model and optimizer
        :param model_state_dict: weights and bias of model
        :param optimizer_state_dict: weight of optimizer
        :param filename: name of the model
        """
        torch.save({
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict
        }, os.path.join(self.config.basic.result_directory, filename))

    def _get_optimizer(self):
        """
        Get different optimizers for different experiments
        :return: Optimizer
        """

        # SGD Optimizer
        if self.config.training.optimizer.type == "SGD":
            return optim.SGD(self.model_device.parameters(),
                             lr=self.config.training.optimizer.learning_rate,
                             momentum=self.config.training.optimizer.momentum)

        # RMSprop Optimizer
        if self.config.training.optimizer.type == "RMSprop":
            return optim.RMSprop(self.model_device.parameters(),
                                 lr=self.config.training.optimizer.learning_rate,
                                 momentum=self.config.training.optimizer.momentum)

        # Adam Optimizer
        if self.config.training.optimizer.type == "Adam":
            return optim.Adam(self.model_device.parameters(),
                              lr=self.config.training.optimizer.learning_rate,
                              betas=(self.config.training.optimizer.beta1, self.config.training.optimizer.beta2),
                              eps=float(self.config.training.optimizer.epsilon),
                              weight_decay=self.config.training.optimizer.weight_decay)

        raise Exception("Optimizer {} does not exist".format(self.config.training.optimizer.type))

    def _get_criterion(self):
        """
        Get different criterions for different experiments
        :return: Loss Function
        """

        # Binary Cross Entropy with Logits Loss
        if self.config.training.criterion.type == "BCEWithLogitsLoss":
            return nn.BCEWithLogitsLoss()

        # Binary Cross Entropy Loss
        if self.config.training.criterion.type == "BCELoss":
            return nn.BCELoss()

        if self.config.training.criterion.type == "CrossEntropyLoss":
            return nn.CrossEntropyLoss()

        raise Exception("Criterion {} does not exist".format(self.config.training.criterion.type))

    def _train3(self):
        # Initialize WandB Reporting if enabled
        if self.config.basic.enable_wand_reporting:
            wandb.init(project="mgs", entity="noahchavannes", name=self.config.basic.result_directory_name,
                       notes=self.config.basic.experiment_description, config=self.config.toDict())

            wandb.watch(self.model_device)

        since = time.time()


        # Structures to save best performing model and optimizer weights
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_opt_wts = copy.deepcopy(self.optimizer.state_dict())
        best_epoch = ''
        best_acc = 0.0

        # Structure to keep track of epoch accuracies and losses of each phase
        epoch_metrics_dict = {
            'train': {
                'accuracy': None,
                'loss': None,
                'tpr': None,
                'tnr': None,
                'acc_eye': None,
                'acc_nose': None,
                'acc_cheeks': None,
                'acc_whiskers': None,
                'acc_ears': None
            },
            'val': {
                'accuracy': None,
                'loss': None,
                'tpr': None,
                'tnr': None,
                'acc_eye': None,
                'acc_nose': None,
                'acc_cheeks': None,
                'acc_whiskers': None,
                'acc_ears': None
            }
        }

        # Training Loop
        for epoch in range(self.config.training.epochs):
            logging.info('Epoch {}/{}'.format(epoch + 1, self.config.training.epochs))

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    # Set model to training mode
                    self.model.train()
                else:
                    # Set model to evaluate mode
                    self.model.eval()

                # Init variables for metrics
                running_loss = 0.0
                correct_classifications = 0
                y_pred = []
                y_true = []
                mgs_attributes_correct = torch.tensor([0,0,0,0,0]).to(self.device)
                # Initialize progress bar
                progress_bar = tqdm(range(self.datasets['dataset_sizes'][phase]))

                # Iterate over data
                for inputs, labels in self.datasets['dataloaders'][phase]:

                    # Update progress bar
                    progress_bar.update(n=inputs.shape[0])

                    # Transfer input and labels to GPU/Device
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # Zero the parameter gradients
                    self.optimizer.zero_grad()

                    # Forward pass
                    # Only track history if in training phase
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get predictions from model
                        predictions = self.model(inputs)

                        # Calculate the loss
                        if self.config.model.mgs_attributes == 1:
                            loss = self.criterion(predictions, labels.float())
                        else:
                            loss = self.criterion(predictions, labels)

                        # Backward pass and optimizer step, only in training phase
                        if phase == 'train':
                            # x = time.time()
                            loss.backward()
                            self.optimizer.step()
                            # y = time.time()
                            # logging.info(f'Backward and optimation: {y-x}') #0.037 - 0.051

                    # TODO: CHECK with train()
                    running_loss += loss.item()

                    if self.config.model.mgs_attributes == 1:
                        # Map probability prediction to yes or no
                        # predictions[predictions < 0.5] = 0
                        # predictions[predictions >= 0.5] = 1
                        # TODO: Per attribute accuracy
                        # torch.sum(p_max_t == l_max_t, dim=0)
                        p = torch.argmax(torch.reshape(predictions, (predictions.shape[0],int(predictions.shape[1]/3),3)), dim=2)
                        l = torch.argmax(torch.reshape(labels, (labels.shape[0],int(labels.shape[1]/3),3)), dim=2)
                        mgs_attributes_correct = torch.add(mgs_attributes_correct, torch.sum(p == l, dim=0))
                        y_pred.extend(p.cpu().numpy())
                        y_true.extend(l.cpu().numpy())
                    else:
                        y_pred.extend(convert_to_label(predictions).cpu())  # Save Prediction
                        labels = labels.data.cpu().numpy()
                        y_true.extend(labels)


                    # Compare label with prediction, sum up correct classifications overall
                    # correct_classifications += torch.sum(predictions == labels.type_as(predictions))

                # Learning rate scheduler step if in validation phase
                if phase == 'val':
                    # learning rate scheduler type is reduce learning rate on plateau
                    if self.config.training.lr_scheduler.type == "ReduceLROnPlateau":
                        # LRScheduler step
                        self.lr_scheduler.step(loss)

                        # If the learning rate is reduced
                        if self.lr_scheduler.in_cooldown:
                            logging.info(
                                "Changed learning rate from {} to {}. Reinitializing model weights with best model from epoch {}".format(
                                    (1 / self.config.training.lr_scheduler.gamma) *
                                    self.optimizer.param_groups[0]["lr"],
                                    self.optimizer.param_groups[0]["lr"],
                                    best_epoch))
                            # Reinitialize the model with the previous best weights
                            self.model_device.load_state_dict(best_model_wts)

                    # learning rate scheduler type is step learning rate
                    else:
                        # LRScheduler step
                        self.lr_scheduler.step()

                # Calculate Epoch Loss
                epoch_loss = running_loss / self.datasets['dataset_sizes'][phase]

                if self.config.model.mgs_attributes == 1:
                    # TODO: Multiclass
                    cf_matrix = confusion_matrix(np.concatenate(y_true).ravel().tolist(), [i for i in np.concatenate(y_pred).ravel().tolist()])
                    fp = cf_matrix.sum(axis=0) - np.diag(cf_matrix)
                    fn = cf_matrix.sum(axis=1) - np.diag(cf_matrix)
                    tp = np.diag(cf_matrix)
                    tn = cf_matrix.sum() - (fp + fn + tp)
                    epoch_mgs_acc = (mgs_attributes_correct / self.datasets['dataset_sizes'][phase]).cpu().numpy().tolist()
                    epoch_metrics_dict[phase]['acc_eye'] = epoch_mgs_acc[0]
                    epoch_metrics_dict[phase]['acc_nose'] = epoch_mgs_acc[1]
                    epoch_metrics_dict[phase]['acc_cheeks'] = epoch_mgs_acc[2]
                    epoch_metrics_dict[phase]['acc_whiskers'] = epoch_mgs_acc[3]
                    epoch_metrics_dict[phase]['acc_ears'] = epoch_mgs_acc[4]
                else:
                    cf_matrix = confusion_matrix(y_true, [i for i in y_pred])
                    tn, fp, fn, tp = cf_matrix.ravel()


                epoch_tpr = tp / (tp + fn)
                epoch_tnr = tn / (tn + fp)
                epoch_accuracy = (tp + tn) / (tp + tn + fp + fn)

                # Save metrics in structure
                if self.config.model.mgs_attributes == 1:
                    epoch_accuracy = epoch_accuracy.mean()
                    epoch_tpr = epoch_tpr.mean()
                    epoch_tnr = epoch_tnr.mean()

                epoch_metrics_dict[phase]['accuracy'] = epoch_accuracy
                epoch_metrics_dict[phase]['loss'] = epoch_loss
                epoch_metrics_dict[phase]['tpr'] = epoch_tpr
                epoch_metrics_dict[phase]['tnr'] = epoch_tnr

                logging.info('{} Loss: {:.4f} \t Acc: {:.4f} \t TPR: {:.4f}  \t TNR: {:.4f}'.format(phase, epoch_loss, epoch_accuracy, epoch_tpr, epoch_tnr))

                # Deep copy the model if new accuracy on validation is better than previous best one
                if phase == 'val' and epoch_accuracy > best_acc:
                    best_acc = epoch_accuracy
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    best_opt_wts = copy.deepcopy(self.optimizer.state_dict())
                    best_epoch = epoch + 1

                # Save Checkpoint during training in predetermined frequency
                if phase == 'val' and (epoch + 1) % self.config.training.save_frequency == 0:
                    self._save_model(copy.deepcopy(self.model.state_dict()), copy.deepcopy(self.optimizer.state_dict()),
                                     '{:03d}.pt'.format(epoch + 1))

            # Report training metrics to wand
            if self.config.basic.enable_wand_reporting:
                # Log current metrics
                wandb.log({
                    "Accuracy Train": epoch_metrics_dict['train']['accuracy'],
                    "Accuracy Val": epoch_metrics_dict['val']['accuracy'],
                    "Loss Train": epoch_metrics_dict['train']['loss'],
                    "Loss Val": epoch_metrics_dict['val']['loss'],
                    "TPR Train": epoch_metrics_dict['train']['tpr'],
                    "TPR Val": epoch_metrics_dict['val']['tpr'],
                    "TNR Train": epoch_metrics_dict['train']['tnr'],
                    "TNR Val": epoch_metrics_dict['val']['tnr']
                }, step=epoch)

                if self.config.model.mgs_attributes == 1:
                    wandb.log({
                        "Accuracy Train Eyes": epoch_metrics_dict['train']['acc_eye'],
                        "Accuracy Val Eyes": epoch_metrics_dict['val']['acc_eye'],
                        "Accuracy Train Nose": epoch_metrics_dict['train']['acc_nose'],
                        "Accuracy Val Nose": epoch_metrics_dict['val']['acc_nose'],
                        "Accuracy Train Cheeks": epoch_metrics_dict['train']['acc_cheeks'],
                        "Accuracy Val Cheeks": epoch_metrics_dict['val']['acc_cheeks'],
                        "Accuracy Train Whiskers": epoch_metrics_dict['train']['acc_whiskers'],
                        "Accuracy Val Whiskers": epoch_metrics_dict['val']['acc_whiskers'],
                        "Accuracy Train Ears": epoch_metrics_dict['train']['acc_ears'],
                        "Accuracy Val Ears": epoch_metrics_dict['val']['acc_ears'],
                    }, step=epoch)
            # Close progress bar
            progress_bar.close()

        # Log training metrics
        time_elapsed = time.time() - since
        logging.info('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        logging.info('Best val Acc: {:4f}'.format(best_acc))

        # Save the best model weights
        self._save_model(best_model_wts, best_opt_wts, 'best-{}.pt'.format(best_epoch))

        # Save latest model
        self._save_model(copy.deepcopy(self.model.state_dict()), copy.deepcopy(self.optimizer.state_dict()),
                         'latest.pt')

        # Reinitialize model with best weights during training and prepare it for returning
        self.model.load_state_dict(best_model_wts)

        # TODO: Check if needed
        if self.config.basic.enable_wand_reporting:
            torch.save(self.model_device.state_dict(), os.path.join(wandb.run.dir, 'model_wand.pt'))

        # Return model
        return self.model


def convert_to_label(output):
    sm = torch.nn.Softmax(dim=1)
    probabilities = sm(output)
    return torch.argmax(probabilities, dim=1)
