'''
Implementation from:

James Allworth, Lloyd Windrim, James Bennett, and Mitch Bryson. A transfer learning approach
to space debris classification using observational light curve data. Acta Astronautica, 181:301–
315, 2021.

'''

import numpy as np
import torch.nn as nn
import torch

class AllworthNet(nn.Module):

    def __init__(self, in_channels, input_size, n_classes):
        super().__init__()
        
        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3, 2, padding=1),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv1d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3, 2, padding=1),
        )


        self.flat = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(64*input_size//4,512),
            nn.Dropout(0.7),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.Dropout(0.7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )
        

    def forward(self, x):
        x = self.cnn1(x.float())
        x = self.cnn2(x)
        x = self.flat(x)
        out = self.fc(x)

        return  out
