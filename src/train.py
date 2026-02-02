import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import os

from . import config

class Train():
    def __init__(self):
        self.config = config.Config()
    
    def trainModel(self, model, trainLoader, validationLoader, epochs = None, learningRate = None, savePath = None, device = None):
        if epochs is None:
            epochs = self.config.epochs
        if learningRate is None:
            learningRate = self.config.learningRate
        if savePath is None:
            savePath = os.path.join(self.config.modelsDirectory, "bestModel.pth")
        if device is None:
            device = self.config.deviceSettings

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr = learningRate)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode = "min",
            patience = self.config.learningRateSchedularPatience,
            factor = self.config.learningRateFactor
        )

        history = {
            "trainLoss": [],
            "trainAcc": [],
            "validationLoss": [],
            "validationAcc": [],
            "learningRates": []
        }
            
        bestValidationAccuracy = 0
        epochsWithoutImprovement = 0
        startTime = time.time()

        print(f"\n🤖 Training for {epochs} epochs...")
        
        for epoch in range(epochs):
            model.train()

            trainingLoss = 0
            trainingCorrect = 0
            trainingTotal = 0

            progressBar = tqdm(trainLoader, desc = f"Epoch {epoch+1}/{epochs}")

            for images, labels in progressBar:
                images = images.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                trainingLoss += loss.item()
                _, predicted = outputs.max(1)
                trainingTotal += labels.size(0)
                trainingCorrect += predicted.eq(labels).sum().item()
                
                progressBar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{100.*trainingCorrect/trainingTotal:.2f}%"
                })

            trainingLoss = trainingLoss / len(trainLoader)
            trainingAcc = trainingCorrect / trainingTotal

            model.eval()

            validationLoss = 0
            validationCorrect = 0
            validationTotal = 0
            
            with torch.no_grad():
                for images, labels in validationLoader:
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    validationLoss += loss.item()
                    _, predicted = outputs.max(1)
                    validationTotal += labels.size(0)
                    validationCorrect += predicted.eq(labels).sum().item()
            
            validationLoss = validationLoss / len(validationLoader)
            validationAcc = validationCorrect / validationTotal

            scheduler.step(validationLoss)
            currentLearningRate = optimizer.param_groups[0]['lr']
            
            if validationAcc > bestValidationAccuracy:
                bestValidationAccuracy = validationAcc
                epochsWithoutImprovement = 0
                
                torch.save({
                    "epoch": epoch,
                    "modelStateDict": model.state_dict(),
                    "validateAcc": validationAcc,
                }, savePath)
                print(f'  ✅ New best model! Val Accuracy -> {validationAcc:.4f}')
            else:
                epochsWithoutImprovement += 1
            
            history["trainLoss"].append(trainingLoss)
            history["trainAcc"].append(trainingAcc)
            history["validationLoss"].append(validationLoss)
            history["validationAcc"].append(validationAcc)
            history["learningRates"].append(currentLearningRate)
            
            print(f'  ⏰ Epoch {epoch+1} -> Train Loss = {trainingLoss:.4f}, '
                f'Train Acc = {trainingAcc:.4f}, Val Acc = {validationAcc:.4f}')
            
            if epochsWithoutImprovement >= self.config.earlyStoppingPatience:
                print(f'  ⚠️ Early stopping -> no improvement for {epochsWithoutImprovement} epochs')
                break
        
        print(f'\n⏰ Training complete in {(time.time() - startTime)/60:.1f} minutes')
        print(f'🏹 Best validation accuracy -> {bestValidationAccuracy:.4f}')
        
        return history
    
    def loadCheckpoint(self, model, checkPointPath, device = None):
        if device is None:
            device = self.config.deviceSettings
            
        checkpoint = torch.load(checkPointPath, map_location = device)
        model.load_state_dict(checkpoint["modelStateDict"])
            
        print(f"⚜️ Loaded model from epoch {checkpoint["epoch"]+1}")
        print(f"🏹 Validation accuracy -> {checkpoint["validateAcc"]:.4f}")
            
        return model, checkpoint