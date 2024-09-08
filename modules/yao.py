'''
Implementation from:

LU Yao and ZHAO Chang-yin. The basic shape classification of space debris with light curves.
Chinese Astronomy and Astrophysics, 45(2):190–208, 2021.

'''

import torch.nn as nn


class YaoNet(nn.Module):

    def __init__(self, in_channels, input_size, n_classes):
        super().__init__()
        
        self.blocks = []
        in_ch = in_channels
        size = input_size

        cnn_layers = []
        for _ in range(3):
            cnn_layers.extend([
                nn.Conv1d(in_channels, 64, 5, padding=2),
                nn.ReLU(),
                nn.Conv1d(64, 64, 5, padding=2),
                nn.Dropout(0.2),
                nn.ReLU(),
                nn.MaxPool1d(3, 2, padding=1)
            ])
            in_channels = 64

        self.cnn = nn.Sequential(*cnn_layers)

        self.flat = nn.Flatten()
        
        self.fc = nn.Sequential(
            nn.Linear(int(64*size / 8),500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, n_classes),
        )

        

    def forward(self, x, features=False):
        x = self.cnn(x)
        x = self.flat(x)
        out = self.fc(x)

        return out
