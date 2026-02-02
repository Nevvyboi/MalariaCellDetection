import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import numpy as np
import os
import typing

from . import config

class Dataset():
    def __init__(self):
        self.trainTransformList = []
        self.config = config.Config()

    def getTransform(self) -> typing.Any:
        if self.config.imageAugmentation["horizontalFlip"]:
            self.trainTransformList.append(transforms.RandomHorizontalFlip(p = 0.5))
        if self.config.imageAugmentation["verticalFlip"]:
            self.trainTransformList.append(transforms.RandomVerticalFlip(p = 0.5))
        if self.config.imageAugmentation["rotationRange"] > 0:
            self.trainTransformList.append(transforms.RandomRotation(degrees = self.config.imageAugmentation["rotationRange"]))
        if self.config.imageAugmentation.get("colorJitter", False):
            self.trainTransformList.append(transforms.ColorJitter(
                brightness = self.config.imageAugmentation.get("brightness", 0.2),
                contrast = self.config.imageAugmentation.get("contrast", 0.2)
            ))

        self.trainTransformList.extend([
            transforms.Resize((self.config.imageSize, self.config.imageSize)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = self.config.normalizeMean, 
                std = self.config.normalizeStandardDeviation
            )
        ])

        trainTransform = transforms.Compose(self.trainTransformList)

        testTransform = transforms.Compose([
            transforms.Resize((self.config.imageSize, self.config.imageSize)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = self.config.normalizeMean, 
                std = self.config.normalizeStandardDeviation
            )
        ])

        return trainTransform, testTransform

    def loadDataset(self, dataDirectory = None, quickMode = False):
        if dataDirectory is None:
            dataDirectory = self.config.dataDirectory
        
        if not os.path.exists(dataDirectory):
            raise FileNotFoundError(f"Dataset not found at {dataDirectory}. Please download from Kaggle and extract to this location.")

        trainTransform, testTransform = self.getTransform()

        fullDataset = datasets.ImageFolder(
            root = dataDirectory,
            transform = trainTransform 
        )
        
        classNames = fullDataset.classes
        
        totalSize = len(fullDataset)
        trainSize = int(self.config.trainSplitRatio * totalSize) 
        valSize = int(self.config.validationSplitRatio * totalSize)  
        testSize = totalSize - trainSize - valSize

        if quickMode:
            subsetSize = min(self.config.quickMode["subsetSize"], totalSize)
            indices = np.random.RandomState(self.config.reproducibilitySeed).permutation(totalSize)[:subsetSize]
            fullDataset = Subset(fullDataset, indices)

            totalSize = subsetSize
            trainSize = int(self.config.trainSplitRatio * totalSize)
            valSize = int(self.config.validationSplitRatio * totalSize)
            testSize = totalSize - trainSize - valSize

            print(f"⚡ Quick mode enabled -> Using subset of {subsetSize} images!")

        trainDataset, validationDataset, testDataset = random_split(
            fullDataset,
            [trainSize, valSize, testSize],
            generator = torch.Generator().manual_seed(self.config.reproducibilitySeed)
        )

        return trainDataset, validationDataset, testDataset, classNames
    
    def createDataLoaders(self, trainDataset, validationDataset, testDataset, batchSize = None) -> tuple:
        if batchSize is None:
            batchSize = self.config.batchSize
        
        trainLoader = DataLoader(
            trainDataset,
            batch_size = batchSize,
            shuffle = True,
            num_workers = self.config.numberOfWorkers,
            pin_memory = self.config.pinMemory,
            drop_last = True
        )

        validationLoader = DataLoader(
            validationDataset,
            batch_size = batchSize,
            shuffle = False,
            num_workers = self.config.numberOfWorkers,
            pin_memory = self.config.pinMemory
        )

        testLoader = DataLoader(
            testDataset,   
            batch_size = batchSize,
            shuffle = False,
            num_workers = self.config.numberOfWorkers,
            pin_memory = self.config.pinMemory
        )

        return trainLoader, validationLoader, testLoader