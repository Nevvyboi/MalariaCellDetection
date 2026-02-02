import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config

class MalariaCellDetection(nn.Module):
    def __init__(self, numOfClasses : int = None):
        super(MalariaCellDetection, self).__init__()
        self.config = config.Config()

        if numOfClasses is None:
            numOfClasses = self.config.numberOfClasses
        
        self.filters = self.config.convolutionalLayersFilters
        
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = self.filters[0], kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(self.filters[0])
        
        self.conv2 = nn.Conv2d(in_channels = self.filters[0], out_channels = self.filters[1], kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(self.filters[1])
        
        self.conv3 = nn.Conv2d(in_channels = self.filters[1], out_channels = self.filters[2], kernel_size = 3, padding = 1)
        self.bn3 = nn.BatchNorm2d(self.filters[2])
        
        self.conv4 = nn.Conv2d(in_channels = self.filters[2], out_channels = self.filters[3], kernel_size = 3, padding = 1)
        self.bn4 = nn.BatchNorm2d(self.filters[3])
        
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.dropoutConv = nn.Dropout2d(self.config.dropoutConvulational)
        self.dropoutFC = nn.Dropout(self.config.dropoutFullyConnected)

        self.finalConnectedInputSize = self.filters[3] * 8 * 8

        self.fc1 = nn.Linear(self.finalConnectedInputSize, self.config.fullyConnectedLayerSizes[0])
        self.fc2 = nn.Linear(self.config.fullyConnectedLayerSizes[0], numOfClasses)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropoutConv(self.pool(F.relu(self.bn2(self.conv2(x)))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropoutConv(self.pool(F.relu(self.bn4(self.conv4(x)))))
        x = x.view(-1, self.finalConnectedInputSize)
        x = self.dropoutFC(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x