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
        
        self.filters = self.config.convolutionalLayersFilters #[32, 64, 128, 256]
        #Block 1 -> 3 -> 32 channels
        self.conv1 = nn.Conv2d(
            in_channels = 3, #RGB images have 3 channels
            out_channels = self.filters[0], #32 filters
            kernel_size = 3, #3x3 filter
            padding = 1 #Preserves spatial dimensions
        )
        self.bn1 = nn.BatchNorm2d(self.filters[0]) #Normalize 32 channels
        #Block 2 -> 32 -> 64 channels
        self.conv2 = nn.Conv2d(
            in_channels = self.filters[0], #32 channels from previous layer
            out_channels = self.filters[1], #64 filters
            kernel_size = 3,
            padding = 1
        )
        self.bn2 = nn.BatchNorm2d(self.filters[1]) #Normalize 64 channels
        #Block 3 -> 64 -> 128 channels
        self.conv3 = nn.Conv2d(
            in_channels = self.filters[1], #64 channels from previous layer
            out_channels = self.filters[2], #128 filters
            kernel_size = 3,
            padding = 1
        )
        self.bn3 = nn.BatchNorm2d(self.filters[2]) #Normalize 128 channels
        #Block 4 -> 128 -> 256 channels
        self.conv4 = nn.Conv2d(
            in_channels = self.filters[2], #128 channels from previous layer 
            out_channels = self.filters[3], #256 filters
            kernel_size = 3,
            padding = 1
        )
        self.bn4 = nn.BatchNorm2d(self.filters[3]) #Normalize 256 channels
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2) #Reduces spatial dimensions by half each time
        self.dropoutConv = nn.Dropout(self.config.dropoutConvulational) #Dropout for every convolutional layer
        self.dropoutFC = nn.Dropout(self.config.dropoutFullyConnected) #Dropout for fully connected layers

        self.finalConnectedInputSize = self.filters[3] * 8 * 8 #256 * 8 * 8 = 16384

        self.fc1 = nn.Linear(self.finalConnectedInputSize, self.config.fullyConnectedLayerSizes[0]) #16384 -> 512
        self.fc2 = nn.Linear(self.config.fullyConnectedLayerSizes[0], numOfClasses) #512 -> 2
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        #Block 1
        x = self.conv1(x)       #(batch, 32, 128, 128)
        x = self.bn1(x)         #Normalize
        x = F.relu(x)           #Activation
        x = self.pool(x)        #(batch, 32, 64, 64)
        
        #Block 2
        x = self.conv2(x)       #(batch, 64, 64, 64)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)        #(batch, 64, 32, 32)
        x = self.dropoutConv(x)
        
        #Block 3
        x = self.conv3(x)       #(batch, 128, 32, 32)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)        #(batch, 128, 16, 16)
        
        #Block 4
        x = self.conv4(x)       #(batch, 256, 16, 16)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool(x)        #(batch, 256, 8, 8)
        x = self.dropoutConv(x)
        
        #Flatten -> (batch, 256, 8, 8) → (batch, 16384)
        x = x.view(-1, self.finalConnectedInputSize)
        
        #Fully connected layers
        x = self.fc1(x)         #(batch, 512)
        x = F.relu(x)
        x = self.dropoutFC(x)
        x = self.fc2(x)         #(batch, 2)
        
        #CrossEntropyLoss applies it internally (more numerically stable)
        
        return x

    def createModel(numberOfClasses = None, device = None):
        if device is None:
            device = config.DEVICE
        
        model = MalariaCellDetection(numOfClasses = numberOfClasses)
        model = model.to(device)  #Move to GPU if available
        
        #Count parameters
        numberOfParameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"🚩 Model created with {numberOfParameters:,} trainable parameters")
        print(f"💻 Device -> {device}")
        
        return model