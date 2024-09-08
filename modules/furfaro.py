'''
Implementation from:

Roberto Furfaro, Richard Linares, and Vishnu Reddy. Space objects classification via light-curve
measurements: deep convolutional neural networks and model-based transfer learning. In AMOS
Technologies Conference, Maui Economic Development Board, pages 1–17, 2018.

'''
import torch.nn as nn
import math

class FurfaroNet(nn.Module):

    def __init__(self, in_channels, input_size, n_classes):
        super().__init__()
        
        cnn_layers = []
        size = input_size
        for stride in [31,11,5]:
            cnn_layers.extend([
                nn.Conv1d(in_channels, 64, stride, padding=stride//2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool1d(3, 2, padding=1)
            ])
            in_channels = 64
            size = math.ceil(size/2)

        self.cnn = nn.Sequential(*cnn_layers)

        self.flat = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(size * 64,500),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(100, n_classes),
        )

    def forward(self, x):
        x = self.cnn(x.float())
        x = self.flat(x)
        out = self.fc(x)
        return out


