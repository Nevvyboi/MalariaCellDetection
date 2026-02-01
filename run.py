#!/usr/bin/env python3
"""
run.py - Main entry point for Malaria Cell Detection CNN

Usage:
    python run.py              # Full training (35 epochs)
    python run.py --quick      # Quick test (3 epochs, small subset)
    python run.py --epochs 50  # Custom number of epochs
    python run.py --evaluate-only  # Skip training, evaluate existing model
"""

import argparse
import os
import sys
import torch
import numpy as np
import random

# Add project root to path
projectRoot = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, projectRoot)

# Import our modules
from src.config import Config
from src.dataset import Dataset
from src.model import MalariaCellDetection
from src.train import Train
from src.evaluate import Evaluate
from src.baseline import Baseline


def setSeed(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parseArguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Malaria Cell Detection CNN")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--quick", action="store_true", help="Quick test mode (3 epochs, small subset)")
    parser.add_argument("--evaluate-only", action="store_true", help="Skip training, evaluate existing model")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline computation")
    return parser.parse_args()


def main():
    args = parseArguments()
    
    # Initialize config
    config = Config()
    
    print("\n" + "=" * 60)
    print("🔬 MALARIA CELL DETECTION WITH CNN")
    print("=" * 60)
    
    # Set seed for reproducibility
    setSeed(config.reproducibilitySeed)
    print(f"\n🎲 Random seed set to {config.reproducibilitySeed}")
    
    # Determine settings based on mode
    if args.quick:
        print("\n⚡ QUICK MODE ENABLED")
        epochs = config.quickMode["epochs"]
        batchSize = config.quickMode["batchSize"]
        print(f"   - Epochs: {epochs}")
        print(f"   - Batch size: {batchSize}")
        print(f"   - Subset size: {config.quickMode['subsetSize']}")
    else:
        epochs = args.epochs if args.epochs else config.epochs
        batchSize = args.batch_size if args.batch_size else config.batchSize
    
    learningRate = args.learning_rate if args.learning_rate else config.learningRate
    
    # =========================================================================
    # STEP 1: Load Dataset
    # =========================================================================
    print("\n" + "-" * 60)
    print("📁 [1/6] LOADING DATASET")
    print("-" * 60)
    
    dataset = Dataset()
    
    try:
        trainData, valData, testData, classNames = dataset.loadDataset(quickMode=args.quick)
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\n📥 Please download the dataset from:")
        print("   https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria")
        print(f"\n📂 Extract to: {config.dataDirectory}")
        sys.exit(1)
    
    trainLoader, valLoader, testLoader = dataset.createDataLoaders(
        trainData, valData, testData, batchSize=batchSize
    )
    
    print(f"\n✅ DataLoaders created (batch_size={batchSize})")
    
    # =========================================================================
    # STEP 2: Create Model
    # =========================================================================
    print("\n" + "-" * 60)
    print("🧠 [2/6] CREATING MODEL")
    print("-" * 60)
    
    model = MalariaCellDetection(numOfClasses=len(classNames))
    model = model.to(config.deviceSettings)
    
    # Count parameters
    numParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🏗️  Model: MalariaCellDetection CNN")
    print(f"📊 Trainable parameters: {numParams:,}")
    print(f"💻 Device: {config.deviceSettings}")
    
    # =========================================================================
    # STEP 3: Train Model
    # =========================================================================
    print("\n" + "-" * 60)
    print("🏋️ [3/6] TRAINING MODEL")
    print("-" * 60)
    
    modelPath = os.path.join(config.modelsDirectory, "bestModel.pth")
    trainer = Train()
    
    if args.evaluate_only:
        print("\n⏭️  Skipping training (--evaluate-only flag)")
        if not os.path.exists(modelPath):
            print(f"\n❌ ERROR: No saved model found at {modelPath}")
            print("   Please train a model first by running without --evaluate-only")
            sys.exit(1)
        history = None
    else:
        print(f"\n🚀 Starting training...")
        print(f"   - Epochs: {epochs}")
        print(f"   - Learning rate: {learningRate}")
        print(f"   - Batch size: {batchSize}")
        
        history = trainer.trainModel(
            model=model,
            trainLoader=trainLoader,
            validationLoader=valLoader,
            epochs=epochs,
            learningRate=learningRate,
            savePath=modelPath,
            device=config.deviceSettings
        )
    
    # Load best model for evaluation
    if os.path.exists(modelPath):
        model, checkpoint = trainer.loadCheckpoint(model, modelPath, config.deviceSettings)
    
    # =========================================================================
    # STEP 4: Evaluate Model
    # =========================================================================
    print("\n" + "-" * 60)
    print("📊 [4/6] EVALUATING MODEL")
    print("-" * 60)
    
    evaluator = Evaluate()
    results = evaluator.evaluateModel(model, testLoader, classNames, config.deviceSettings)
    
    # =========================================================================
    # STEP 5: Compute Baseline
    # =========================================================================
    print("\n" + "-" * 60)
    print("📉 [5/6] COMPUTING BASELINE")
    print("-" * 60)
    
    if args.skip_baseline:
        print("\n⏭️  Skipping baseline (--skip-baseline flag)")
        baselineResults = None
    else:
        baseline = Baseline()
        maxSamples = 2000 if args.quick else 5000
        baselineResults = baseline.computeBaseline(trainData, testData, classNames, maxSamples)
        
        # Print comparison
        print("\n" + "=" * 50)
        print("📈 MODEL COMPARISON")
        print("=" * 50)
        print(f"{'Metric':<15} {'Baseline':<12} {'CNN':<12} {'Improvement':<12}")
        print("-" * 51)
        print(f"{'Accuracy':<15} {baselineResults['accuracy']:<12.4f} {results['accuracy']:<12.4f} +{results['accuracy']-baselineResults['accuracy']:<11.4f}")
        print(f"{'F1 Score':<15} {baselineResults['f1']:<12.4f} {results['f1']:<12.4f} +{results['f1']-baselineResults['f1']:<11.4f}")
    
    # =========================================================================
    # STEP 6: Generate Visualizations
    # =========================================================================
    print("\n" + "-" * 60)
    print("📊 [6/6] GENERATING VISUALIZATIONS")
    print("-" * 60)
    
    # Training history plot
    if history:
        historyPath = os.path.join(config.outputsDirectory, "training_history.png")
        evaluator.plotTrainingHistory(history, savePath=historyPath)
    
    # Confusion matrix
    cmPath = os.path.join(config.outputsDirectory, "confusion_matrix.png")
    evaluator.plotConfusionMatrix(
        results["labels"], 
        results["predictions"], 
        classNames, 
        savePath=cmPath
    )
    
    # ROC curve
    rocPath = os.path.join(config.outputsDirectory, "roc_curve.png")
    evaluator.plotRocCurve(
        results["labels"],
        results["probabilities"],
        savePath=rocPath
    )
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n📊 Final Results:")
    print(f"   Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"   Precision: {results['precision']:.4f}")
    print(f"   Recall:    {results['recall']:.4f}")
    print(f"   F1 Score:  {results['f1']:.4f}")
    print(f"   ROC-AUC:   {results['roc_auc']:.4f}")
    
    if baselineResults:
        improvement = results['accuracy'] - baselineResults['accuracy']
        print(f"\n📈 Improvement over baseline: +{improvement:.4f} (+{improvement*100:.2f}%)")
    
    print(f"\n📁 Outputs saved to: {config.outputsDirectory}/")
    print(f"   - training_history.png")
    print(f"   - confusion_matrix.png")
    print(f"   - roc_curve.png")
    
    print(f"\n💾 Best model saved to: {modelPath}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()