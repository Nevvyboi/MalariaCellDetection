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

    def getTransform(self) -> typing.Any: #Defines image tranformation pipelines
        if self.config.imageAugmentation["horizontalFlip"]:
            self.trainTransformList.append(transforms.RandomHorizontalFlip(p = 0.5)) #p = 0.5 means 50% chance of flipping image
        if self.config.imageAugmentation["verticalFlip"]:
            self.trainTransformList.append(transforms.RandomVerticalFlip(p = 0.5)) #p = 0.5 means 50% chance of flipping image
        if self.config.imageAugmentation["rotationRange"] > 0:
            self.trainTransformList.append(transforms.RandomRotation(degrees = self.config.imageAugmentation["rotationRange"])) #Rotates image randomly within -20 degrees and 20 degrees
        if self.config.imageAugmentation.get("colorJitter", False):
            self.trainTransformList.append(transforms.ColorJitter(
                brightness = self.config.imageAugmentation.get("brightness", 0.2),
                contrast = self.config.imageAugmentation.get("contrast", 0.2)
                )
            )

        self.trainTransformList.extend([
            transforms.Resize((self.config.imageSize, self.config.imageSize)), #Resizes image to 128 x 128 pixels (consistent image sizes)
            transforms.ToTensor(), #Convert image to tensor -> scae 0 to 1
            transforms.Normalize( #Normalize with ImageNet stats
                mean = self.config.normalizeMean, 
                std = self.config.normalizeStandardDeviation
                )
        ])

        trainTransform = transforms.Compose(self.trainTransformList) #Chains all transformations together
        #Order of transformations matters: image -> flip? -> rotate? -> resize -> tensor -> normalize -> output

        testTransform = transforms.Compose([ #No augmentation for validation and testing
            transforms.Resize((self.config.imageSize, self.config.imageSize)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = self.config.normalizeMean, 
                std = self.config.normalizeStandardDeviation
                )
        ]) #No augmentation, so we can test on real images. Augmentation is only for training data/variety

        return trainTransform, testTransform

    def loadDataset(self, dataDirectory = None, quickMode = False) -> None: #Loads the malaria images dataset into training/validation/testing sets
        if dataDirectory is None:
            dataDirectory = self.config.dataDirectory
        
        if not os.path.exists(dataDirectory):
            raise FileNotFoundError(f"Dataset not found at {dataDirectory}. Please download from Kaggle and extract to this location.")

        trainTransform, testTransform = self.getTransform()

        fullDataset = datasets.ImageFolder( #Scans folder structure and uses folder names as class labels; thus creating a (image, label) pair
            root = dataDirectory,
            transform = trainTransform 
        )
        
        classNames = fullDataset.classes #Will get ["Infected", "Uninfected"]

        print(f"✅ Loaded {len(fullDataset)} images!")
        print(f"🔰  Classes: {classNames}")
        
        totalSize = len(fullDataset) #Calculating split set sizes
        trainSize = int(config.trainSplitRatio * totalSize) # trainSize = 0.7 * 27 558 = 19 290
        valSize = int(config.validationSplitRatio * totalSize) # valSize = 0.15 * 27 558 = 4 133
        testSize = totalSize - trainSize - valSize # testSize = 27 558 - 19 290 - 4 133 = 4 135

        if quickMode:
            subsetSize = min(self.config.quickMode["subsetSize"], totalSize) #Use smaller subset for quick testing
            indices = np.random.RandomState(self.config.reproducibilitySeed).permutation(totalSize)[:subsetSize] #Randomly select subsetSize indices
            fullDataset = Subset(fullDataset, indices)

            totalSize = subsetSize
            trainSize = int(self.config.trainSplitRatio * totalSize)
            valSize = int(self.config.validationSplitRatio * totalSize)
            testSize = totalSize - trainSize - valSize

            print(f"⚡ Quick mode enabled -> Using subset of {subsetSize} images!")

        trainDataset, validationDataset, testDataset = random_split( #Splitting dataset into training, validation and testing sets
            fullDataset,
            [trainSize, valSize, testSize],
            generator = torch.Generator().manual_seed(self.config.reproducibilitySeed) #Ensures reproducible splits
        )

        print(f"🚂 Train: {len(trainDataset)} | 🧪 Val: {len(validationDataset)} | 🧐 Test: {len(testDataset)}")

        return trainDataset, validationDataset, testDataset, classNames
    
    def createDataLoaders(self, trainDataset, validationDataset, testDataset, batchSize = None) -> tuple[DataLoader, DataLoader, DataLoader]:
        if batchSize is None:
            batchSize = self.config.batchSize
        
        trainLoader = DataLoader(
            trainDataset,
            batch_size = batchSize,
            shuffle = True, #Shuffles data every epoch to prevent model from learning order
            num_workers = self.config.numberOfWorkers,
            pin_memory = self.config.pinMemory,
            drop_last = True #Drops last incomplete batch if dataset size is not divisible by batch size
        )

        validationLoader = DataLoader(
            validationDataset,
            batch_size = batchSize,
            shuffle = False, #No need to shuffle validation data
            num_workers = self.config.numberOfWorkers,
            pin_memory = self.config.pinMemory
        )

        testLoader = DataLoader(
            testDataset,   
            batch_size = batchSize,
            shuffle = False, #No need to shuffle test data
            num_workers = self.config.numberOfWorkers,
            pin_memory = self.config.pinMemory
        )

        return trainLoader, validationLoader, testLoader



