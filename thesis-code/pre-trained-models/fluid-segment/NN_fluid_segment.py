"""
File for class defintion (neural net architecture)
of fluid segmentation trained on velocity data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # First fully connected layer
        self.fc1 = nn.Linear(3, 80)
        # Second fully connected layer
        self.fc2 = nn.Linear(80, 40)
        # third fully connected layer
        self.fc3 = nn.Linear(40, 10)
        # fourth fully connected layer
        self.fc4 = nn.Linear(10, 2)

    # x represents our data
    def forward(self, x):
        compo = x
        compo = self.fc1(compo)
        compo = self.fc2(compo)
        compo = self.fc3(compo)
        compo = torch.tanh(compo)
        compo = self.fc4(compo)

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
