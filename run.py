#Nevin Tom -> -> CSO7013 Machine Learning 2517238
"""
run.py - Main entry point for Malaria Cell Detection CNN

Usage:
    python run.py                    # Full training (35 epochs)
    python run.py --quick            # Quick test (3 epochs, small subset)
    python run.py --epochs 50        # Custom number of epochs
    python run.py --evaluate-only    # Skip training, evaluate existing model
    python run.py --predict IMAGE    # Predict on a specific image
    python run.py --predict-random   # Predict on random test image
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import random

#Adding project root to path
projectRoot = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, projectRoot)

#Importing our modules
from src.config import Config
from src.dataset import Dataset
from src.model import MalariaCellDetection
from src.train import Train
from src.evaluate import Evaluate
from src.baseline import Baseline


def setSeed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parseArguments():
    parser = argparse.ArgumentParser(description = "Malaria Cell Detection CNN")
    parser.add_argument("--epochs", type = int, default = None, help = "Number of training epochs")
    parser.add_argument("--batch-size", type = int, default = None, help = "Batch size")
    parser.add_argument("--learning-rate", type = float, default = None, help = "Learning rate")
    parser.add_argument("--quick", action = "store_true", help = "Quick test mode (3 epochs, small subset)")
    parser.add_argument("--evaluate-only", action = "store_true", help = "Skip training, evaluate existing model")
    parser.add_argument("--skip-baseline", action = "store_true", help = "Skip baseline computation")
    parser.add_argument("--predict", type = str, default = None, help = "Predict on a specific image (provide path)")
    parser.add_argument("--predict-random", action = "store_true", help = "Predict on a random test image")
    return parser.parse_args()


def printHeader():
    print("\n")
    print("🔬 MALARIA CELL DETECTION WITH CNN 🔬            ")
    print("Detecting malaria-infected cells from microscope images  ")


def printSection(number, title, icon = "📌"):
    print(f"\n{icon} [{number}/6] {title:<47}\n")


def predictSingleImage(model, imagePath, classNames, config, device):
    from PIL import Image
    from torchvision import transforms
    
    print("\n")
    print("🎯 SINGLE IMAGE PREDICTION")
    print("Analyzing cell image for malaria detection")
    
    #Loading image
    image = Image.open(imagePath).convert("RGB")
    originalSize = image.size
    
    print(f"\n   📷 Image Details")
    print(f"   ├── Path -> {imagePath}")
    print(f"   ├── Original size -> {originalSize[0]}×{originalSize[1]}")
    print(f"   └── Mode -> RGB")
    
    #Transforming image
    transform = transforms.Compose([
        transforms.Resize((config.imageSize, config.imageSize)),
        transforms.ToTensor(),
        transforms.Normalize(mean = config.normalizeMean, std = config.normalizeStandardDeviation)
    ])
    
    inputTensor = transform(image).unsqueeze(0).to(device)
    
    print(f"\n   ⚙️  Preprocessing")
    print(f"   ├── Resized to -> {config.imageSize}×{config.imageSize}")
    print(f"   ├── Normalized -> ImageNet stats")
    print(f"   └── Device -> {device}")
    
    #Predicting
    model.eval()
    with torch.no_grad():
        outputs = model(inputTensor)
        probabilities = F.softmax(outputs, dim = 1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predictedClass = classNames[predicted.item()]
    confidenceScore = confidence.item() * 100
    allProbs = probabilities.squeeze().cpu().numpy()
    
    print(f"\n   🎯 Prediction Results")
    print(f"   ├── Predicted Class -> {predictedClass}")
    print(f"   ├── Confidence -> {confidenceScore:.2f}%")
    print(f"   │")
    print(f"   └── Class Probabilities:")
    for i, className in enumerate(classNames):
        bar = "█" * int(allProbs[i] * 25) + "░" * (25 - int(allProbs[i] * 25))
        marker = " ◄ PREDICTED" if i == predicted.item() else ""
        print(f"       ├── {className:<12} -> {allProbs[i]*100:6.2f}% │{bar}│{marker}")
    
    print(f"\n   🩺 Diagnosis")
    if predictedClass == "Parasitized":
        print(f"   ├── Result -> 🦠 MALARIA DETECTED")
        print(f"   └── Note -> This cell appears to be infected with malaria parasites")
    else:
        print(f"   ├── Result -> ✅ HEALTHY CELL")
        print(f"   └── Note -> This cell appears to be uninfected")
    
    print("\n")
    
    return predictedClass, confidenceScore


def predictRandomImage(model, testLoader, classNames, config, device):
    print("\n")
    print("🎲 RANDOM TEST IMAGE PREDICTION")
    print("Selecting random image from test dataset")
    
    #Getting random batch and selecting random image
    dataIter = iter(testLoader)
    images, labels = next(dataIter)
    
    idx = np.random.randint(0, len(images))
    image = images[idx].unsqueeze(0).to(device)
    trueLabel = labels[idx].item()
    
    print(f"\n   📷 Image Details")
    print(f"   ├── Source -> Test dataset (index {idx})")
    print(f"   ├── True Label -> {classNames[trueLabel]}")
    print(f"   ├── Size -> {config.imageSize}×{config.imageSize}")
    print(f"   └── Device -> {device}")
    
    #Predicting
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim = 1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predictedClass = classNames[predicted.item()]
    confidenceScore = confidence.item() * 100
    trueClass = classNames[trueLabel]
    allProbs = probabilities.squeeze().cpu().numpy()
    
    print(f"\n   🎯 Prediction Results")
    print(f"   ├── True Class -> {trueClass}")
    print(f"   ├── Predicted Class -> {predictedClass}")
    print(f"   ├── Confidence -> {confidenceScore:.2f}%")
    print(f"   │")
    print(f"   └── Class Probabilities:")
    for i, className in enumerate(classNames):
        bar = "█" * int(allProbs[i] * 25) + "░" * (25 - int(allProbs[i] * 25))
        marker = " ◄ PREDICTED" if i == predicted.item() else ""
        print(f"       ├── {className:<12} -> {allProbs[i]*100:6.2f}% │{bar}│{marker}")
    
    print(f"\n   📋 Evaluation")
    if predictedClass == trueClass:
        print(f"   ├── Status -> ✅ CORRECT PREDICTION")
        print(f"   └── The model correctly identified this as {predictedClass}")
    else:
        print(f"   ├── Status -> ❌ INCORRECT PREDICTION")
        print(f"   ├── Expected -> {trueClass}")
        print(f"   └── Got -> {predictedClass}")
    
    print("\n")
    
    return predictedClass, trueClass, confidenceScore


def main():
    args = parseArguments()
    
    #Initialising config
    config = Config()
    
    #Printing the header
    printHeader()
    
    #Setting seed for reproducibility
    setSeed(config.reproducibilitySeed)
    print(f"\n⚙️  Configuration")
    print(f"   ├── Seed -> {config.reproducibilitySeed}")
    print(f"   └── Device -> {config.deviceSettings}")
    
    #Determining settings based on mode
    if args.quick:
        epochs = config.quickMode["epochs"]
        batchSize = config.quickMode["batchSize"]
        print(f"\n⚡ Quick Mode")
        print(f"   ├── Epochs -> {epochs}")
        print(f"   ├── Batch size -> {batchSize}")
        print(f"   └── Subset -> {config.quickMode['subsetSize']} images")
    else:
        epochs = args.epochs if args.epochs else config.epochs
        batchSize = args.batch_size if args.batch_size else config.batchSize
    
    learningRate = args.learning_rate if args.learning_rate else config.learningRate
    
    #Check if prediction mode (need minimal setup)
    isPredictMode = args.predict or args.predict_random
    
    printSection("1", "LOADING DATASET", "📁")
    
    dataset = Dataset()
    
    try:
        trainData, valData, testData, classNames = dataset.loadDataset(quickMode = args.quick or isPredictMode)
    except FileNotFoundError as e:
        print(f"\n   ❌ ERROR: {e}")
        print(f"\n   📥 Download dataset from:")
        print(f"      https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria")
        print(f"\n   📂 Extract to: {config.dataDirectory}")
        sys.exit(1)
    
    trainLoader, valLoader, testLoader = dataset.createDataLoaders(
        trainData, 
        valData, 
        testData, 
        batchSize = batchSize
    )
    
    print(f"\n   📊 Dataset Summary")
    print(f"   ├── Total images -> {len(trainData) + len(valData) + len(testData):,}")
    print(f"   ├── Classes -> {classNames}")
    print(f"   ├── Training set -> {len(trainData):,} images")
    print(f"   ├── Validation set -> {len(valData):,} images")
    print(f"   ├── Test set -> {len(testData):,} images")
    print(f"   └── Batch size -> {batchSize}")
    
    printSection("2", "CREATING MODEL", "🧠")
    
    model = MalariaCellDetection(numOfClasses = len(classNames))
    model = model.to(config.deviceSettings)
    
    #Counting number of  parameters
    numParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n   🏗️  Model Architecture")
    print(f"   ├── Name: MalariaCellDetection CNN")
    print(f"   ├── Conv layers: {len(config.convolutionalLayersFilters)} blocks")
    print(f"   │   └── Filters: {config.convolutionalLayersFilters}")
    print(f"   ├── FC layers: {config.fullyConnectedLayerSizes}")
    print(f"   ├── Dropout: Conv = {config.dropoutConvulational}, FC = {config.dropoutFullyConnected}")
    print(f"   ├── Parameters: {numParams:,}")
    print(f"   └── Device: {config.deviceSettings}")
    
    printSection("3", "TRAINING MODEL", "🏋️")
    
    modelPath = os.path.join(config.modelsDirectory, "bestModel.pth")
    trainer = Train()
    
    if args.evaluate_only or isPredictMode:
        print(f"\n   ⏭️  Loading existing model...")
        if not os.path.exists(modelPath):
            print(f"\n   ❌ ERROR: No saved model at {modelPath}")
            print(f"   Please run training first: python run.py --quick")
            sys.exit(1)
        history = None
        model, checkpoint = trainer.loadCheckpoint(model, modelPath, config.deviceSettings)
    else:
        print(f"\n   🚀 Training Configuration")
        print(f"   ├── Epochs -> {epochs}")
        print(f"   ├── Learning rate -> {learningRate}")
        print(f"   ├── Batch size -> {batchSize}")
        print(f"   ├── Optimizer -> Adam")
        print(f"   ├── Loss -> CrossEntropyLoss")
        print(f"   ├── LR Scheduler -> ReduceLROnPlateau (patience = {config.learningRateSchedularPatience})")
        print(f"   └── Early stopping -> {config.earlyStoppingPatience} epochs\n")
        
        history = trainer.trainModel(
            model = model,
            trainLoader = trainLoader,
            validationLoader = valLoader,
            epochs = epochs,
            learningRate = learningRate,
            savePath = modelPath,
            device = config.deviceSettings
        )
        
        #Load best model for evaluation
        if os.path.exists(modelPath):
            model, checkpoint = trainer.loadCheckpoint(model, modelPath, config.deviceSettings)
    
    #Handle prediction modes
    if args.predict:
        if not os.path.exists(args.predict):
            print(f"\n   ❌ ERROR: Image not found at {args.predict}")
            sys.exit(1)
        predictSingleImage(model, args.predict, classNames, config, config.deviceSettings)
        return
    
    if args.predict_random:
        predictRandomImage(model, testLoader, classNames, config, config.deviceSettings)
        return
    
    printSection("4", "EVALUATING MODEL", "📊")
    
    evaluator = Evaluate()
    results = evaluator.evaluateModel(model, testLoader, classNames, config.deviceSettings)
    
    printSection("5", "COMPUTING BASELINE", "📉")
    
    if args.skip_baseline:
        print(f"\n   ⏭️  Skipping baseline (--skip-baseline)")
        baselineResults = None
    else:
        baseline = Baseline()
        maxSamples = 2000 if args.quick else 5000
        baselineResults = baseline.computeBaseline(trainData, testData, classNames, maxSamples)
    
    printSection("6", "GENERATING VISUALIZATIONS", "📈")
    
    #Training history plot
    if history:
        historyPath = os.path.join(config.outputsDirectory, "trainingHistory.png")
        evaluator.plotTrainingHistory(history, savePath = historyPath)
    
    #Confusion matrix
    cmPath = os.path.join(config.outputsDirectory, "confusionMatrix.png")
    evaluator.plotConfusionMatrix(
        results["labels"], 
        results["predictions"], 
        classNames, 
        savePath = cmPath
    )
    
    #ROC curve
    rocPath = os.path.join(config.outputsDirectory, "rocCurve.png")
    evaluator.plotRocCurve(
        results["labels"],
        results["probabilities"],
        savePath = rocPath
    )
    
    print(f"\n   📁 Saved to {config.outputsDirectory}/")
    print(f"   ├── trainingHistory.png")
    print(f"   ├── confusionMatrix.png")
    print(f"   └── rocCurve.png")
    
    print("\n")
    print("✅ COMPLETE")
    
    print(f"\n   📊 Final Results")
    print(f"   ├── Accuracy -> {results["accuracy"]:.4f} ({results["accuracy"] * 100:.2f}%)")
    print(f"   ├── Precision -> {results["precision"]:.4f}")
    print(f"   ├── Recall -> {results["recall"]:.4f}")
    print(f"   ├── F1 Score -> {results["f1"]:.4f}")
    print(f"   └── ROC-AUC -> {results["roc_auc"]:.4f}")
    
    if baselineResults:
        print(f"\n   📈 Comparison")
        print(f"   ├── Baseline Accuracy -> {baselineResults["accuracy"]:.4f}")
        print(f"   ├── CNN Accuracy: -> {results["accuracy"]:.4f}")
        print(f"   └── Improvement -> +{(results["accuracy"] - baselineResults["accuracy"]) * 100:.2f}%")
    
    print(f"\n   💾 Model saved: {modelPath}")
    print("\n")


if __name__ == "__main__":
    main()