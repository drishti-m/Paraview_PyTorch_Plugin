"""
File for class defintion (neural net architecture)
of fluid classifier trained on pressure data.
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        batch_size = 2500
        # First fully connected layer
        self.fc1 = nn.Linear(batch_size, 50)
        # Second fully connected layer
        self.fc2 = nn.Linear(50, 20)
        # Third fully connected layer
        self.fc3 = nn.Linear(20, 2)

    # x represents our data
    def forward(self, x):
        compo = x
        compo = self.fc1(compo)
        compo = self.fc2(compo)
        compo = torch.tanh(compo)
        compo = self.fc3(compo)
        return compo

    def predict(self, x):
        pred = F.softmax(self.forward(x), dim=1)
        ans = []
        # Pick the class with maximum weight
        for t in pred:
            if t[0] > t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)
