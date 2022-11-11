import torch
from torch import nn


class Network(nn.Module):
    
    def __init__(self, output_dim):
        super(Network, self).__init__()

        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        
        # image size: 28 * 28
        self.cnn_lay = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
    
            nn.Conv2d(32, 64, 3, 1, 0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
    
            nn.Conv2d(64, 256, 3, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0)
        )
        
        self.net = nn.Sequential(
            nn.Linear(256 * 25, 1024),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        
    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.cnn_lay(x)
        x = x.flatten(1)
        
        return self.net(x)
