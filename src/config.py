import torch
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Config():
    def __init__(self):

        self.dataDirectory = os.path.join(root, 'data', 'cellImages')
        self.modelsDirectory = os.path.join(root, 'models')
        self.outputsDirectory = os.path.join(root, 'outputs')

        self.reproducibilitySeed = 42

        self.trainSplitRatio = 0.7 #70% for training
        self.validationSplitRatio = 0.15 #15% for validation
        self.testSplitRatio = 0.15 #15% for testing

        self.imageSize = 128 #This will resize all data images to 128 x 128 pixels
        self.numberOfChannels = 3 #RGB images have 3 channels
        self.numberOfClasses = 2 #Number of cell classifications

        self.classificationOptions = ["Infected", "Uninfected"]

        self.normalizeMean = [0.485, 0.456, 0.406] #Mean for each channel
        self.normalizeStandardDeviation = [0.229, 0.224, 0.225] #Standard deviation for each channel

        self.batchSize = 64 #Number of images per training step
        self.epochs = 35 #Number of complete training passes through the dataset
        self.learningRate = 0.001 #Step size for weight updates

        self.learningRateSchedularPatience = 3 #Number of epochs with no improvement after which learning rate will be reduced
        self.learningRateFactor = 0.5 #Multiplier for reducing learning rate

        self.earlyStoppingPatience = 7 #Number of epochs with no improvement after which training will be stopped

        self.convolutionalLayersFilters = [32, 64, 128, 256] #Number of filters in each convolutional layer
        self.fullyConnectedLayerSizes = [512] #Number of neurons in each fully connected layer

        self.dropoutConvulational = 0.25 #25% dropout rate for convolutional layers
        self.dropoutFullyConnected = 0.5 #50% dropout rate for fully connected layers

        self.imageAugmentation = {
            "horizontalFlip" : True, #Flips images left right
            "verticalFlip" : True, #Flips images up down
            "rotationRange" : 20, #Rotates image 20 degrees randomly
            "colorJitter" : True, #Varies brightness, contrast and saturation
            "brightness" : 0.2, #Brightness variation factor
            "contrast" : 0.2, #Contrast variation factor
        }  

        self.deviceSettings = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Uses GPI if available otherwises falls back to cpu
        self.numberOfWorkers = 4 if os.name != 'nt' else 0 #Number of subprocesses for data loading, 0 for Windows
        self.pinMemory = torch.cuda.is_available() #Speeds up data transfer to GPU

        self.quickMode = { #Quick testing if code works without full training
            "epochs" : 3,
            "batchSize" : 32,
            "subsetSize" : 2000
        }

        self.createDirectories()

    def createDirectories(self):
        os.makedirs(self.modelsDirectory, exist_ok = True)
        os.makedirs(self.outputsDirectory, exist_ok = True)
