# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPerceptron(nn.Module):
    """ a simple perceptron based classifier """
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Args:
            input_dim (int): the size of the input feature vector
            hidden_dim (int): the size of the hidden feature vector
            output_dim (int): the size of the output feature vector
        """
        super(MLPerceptron, self).__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, x_in, apply_softmax=False):
        """The forward pass of the classifier

        Args:
            x_in (torch.Tensor): an input data tensor.
                x_in.shape should be (batch, input_dim)
            apply_sigmoid (bool): a flag for the sigmoid activation
                should be false if used with the Cross Entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch,)
        """
        intermediate_vector = F.relu(self.fc1(x_in))

        prediciton_vector = self.fc2(F.dropout(intermediate_vector, p=0.2))

        if apply_softmax:
            prediciton_vector = F.softmax(prediciton_vector, dim=1) #

        return prediciton_vector
