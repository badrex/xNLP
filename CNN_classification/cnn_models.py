import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_classifier(nn.Module):
    def __init__(self, initial_num_channels, num_classes, num_channels):
        """
        Args:
            initial_num_channels (int): size of the incoming feature vector
            num_classes (int): size of the output prediciton vector
            num_channels (int): constant channel size to use throughout NN
        """
        super(CNN_classifier, self).__init__()

        self.convnet = nn.Sequential(
            nn.Conv1d(in_channels=initial_num_channels,
                out_channels=num_channels, kernel_size=3),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels,
                out_channels=num_channels, kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels,
                out_channels=num_channels, kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels,
                out_channels=num_channels, kernel_size=3),
            nn.ELU()
        )
        self.fc = nn.Linear(num_channels, num_classes)


    def forward(self, x_surname, apply_softmax=False):
        """ The forward pass of the CNN_classifier

        Args:
            x_surname (torch.Tensor): an input data tensor
                x_surname.shape should be (batch, initial_num_channels,
                    max_surname_length)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the cross-entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, num_classes)
        """

        features = self.convnet(x_surname).squeeze(dim=2)
        prediciton_vector = self.fc(features)

        if apply_softmax:
            prediciton_vector = F.softmax(prediciton_vector, dim=1)

        return prediciton_vector
