import typing
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from tqdm import tqdm
import os

from . import config

class Evaluate():
    def __init__(self):
        self.config = config.Config()
    
    def evaluateModel(self, model, testLoader, classNames, device = None) -> typing.Dict[str, float]:
        if device is None:
            device = self.config.deviceSettings
    
        model.eval()

        allPredictions = []
        allLabels = []
        allProbabilities = []

        with torch.no_grad():
            for images, labels in tqdm(testLoader, desc = "Evaluating"):
                images = images.to(device)
                outputs = model(images)
                probabilities = F.softmax(outputs, dim = 1)
                _, predicted = outputs.max(1)
                
                allPredictions.extend(predicted.cpu().numpy())
                allLabels.extend(labels.numpy())
                allProbabilities.extend(probabilities.cpu().numpy())
        
        allPredictions = np.array(allPredictions)
        allLabels = np.array(allLabels)
        allProbabilities = np.array(allProbabilities)

        accuracy = accuracy_score(allLabels, allPredictions)
        precision = precision_score(allLabels, allPredictions)
        recall = recall_score(allLabels, allPredictions)
        f1 = f1_score(allLabels, allPredictions)
        rocAuc = roc_auc_score(allLabels, allProbabilities[:, 1])

        print("\n" + "=" * 50)
        print("📄 TEST RESULTS")
        print("=" * 50)
        print(f"Accuracy ->  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision -> {precision:.4f}")
        print(f"Recall ->    {recall:.4f}")
        print(f"F1 Score ->  {f1:.4f}")
        print(f"ROC-AUC ->   {rocAuc:.4f}")
        print("\n" + classification_report(allLabels, allPredictions, target_names = classNames))

        return {
            "accuracy": accuracy, "precision": precision, "recall": recall,
            "f1": f1, "roc_auc": rocAuc, "predictions": allPredictions,
            "labels": allLabels, "probabilities": allProbabilities
        }

    def plotTrainingHistory(self, history, savePath = None) -> None:
        fig, axes = plt.subplots(1, 3, figsize = (15, 4))
        epochs = range(1, len(history['trainLoss']) + 1)
        
        axes[0].plot(epochs, history['trainLoss'], "b-", label = "Train")
        axes[0].plot(epochs, history['validationLoss'], "r-", label = "Validation")
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss over Time'); axes[0].legend(); axes[0].grid(True)
        
        axes[1].plot(epochs, history['trainAcc'], "b-", label = "Train")
        axes[1].plot(epochs, history['validationAcc'], "r-", label = "Validation")
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy over Time'); axes[1].legend(); axes[1].grid(True)
        
        axes[2].plot(epochs, history['learningRates'], "g-")
        axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule'); axes[2].grid(True)
        
        plt.tight_layout()
        if savePath:
            plt.savefig(savePath, dpi = 150, bbox_inches = "tight")
            print(f"📃 Saved -> {savePath}")
        plt.close()
    
    def plotConfusionMatrix(self, labels, predictions, classNames, savePath = None) -> None:
        cm = confusion_matrix(labels, predictions)
        plt.figure(figsize = (8, 6))
        sns.heatmap(cm, annot = True, fmt = "d", cmap="Blues", xticklabels = classNames, yticklabels = classNames)
        plt.xlabel('Predicted'); plt.ylabel('Actual')
        plt.title('Confusion Matrix'); plt.tight_layout()
        if savePath:
            plt.savefig(savePath, dpi = 150, bbox_inches = "tight")
            print(f"📃 Saved -> {savePath}")
        plt.close()
    
    def plotRocCurve(self, labels, probabilities, savePath = None) -> None:  # FIXED: added self
        fpr, tpr, _ = roc_curve(labels, probabilities[:, 1])
        auc = roc_auc_score(labels, probabilities[:, 1])
        plt.figure(figsize = (8, 6))
        plt.plot(fpr, tpr, "b-", linewidth = 2, label = f"CNN (AUC = {auc:.4f})")
        plt.plot([0, 1], [0, 1], "k--", label = "Random Guessing")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("ROC Curve"); plt.legend(); plt.grid(True); plt.tight_layout()
        if savePath:
            plt.savefig(savePath, dpi = 150, bbox_inches = "tight")
            print(f"📃 Saved -> {savePath}")
        plt.close()