import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  #Displaying progress bars
import time
import os

from . import config

"""
For each EPOCH (complete pass through data):
│
├─► TRAINING PHASE
│   │
│   └─► For each BATCH of 64 images:
│       │
│       ├── 1. Zero gradients (clear from last batch)
│       ├── 2. Forward pass (get predictions)
│       ├── 3. Compute loss (how wrong are we?)
│       ├── 4. Backward pass (compute gradients)
│       └── 5. Update weights (improve model)
│
├─► VALIDATION PHASE
│   │
│   └─► For each batch (no learning, just measuring):
│       ├── Forward pass
│       └── Compute accuracy
│
├─► Adjust learning rate if needed
│
└─► Save model if it's the best so far
"""

class Train():
    def __init__(self):
        self.config = config.Config()
    
    def trainModel(self, model, trainLoder, validationLoader, epochs = None, learningRate = None, savePath = None, device = None) -> None:
        if epochs is None:
            epochs = self.config.epochs
        if learningRate is None:
            learningRate = self.config.learningRate
        if savePath is None:
            savePath = os.path.join(self.config.modelsDirectory, 'bestModel.pth')
        if device is None:
            device = self.config.deviceSettings

        criterion = nn.CrossEntropyLoss() #Suitable for multi-class classification -> It combines softmax + negative log likelihood
        optimizer = optim.Adam(model.parameters(), lr = learningRate) #Adaptive learning rate optimization algorithm

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode = "min",                  #Reduce when val_loss stops decreasing
            patience = self.config.learningRateSchedularPatience, #Wait 3 epochs before reducing
            factor = self.config.learningRateFactor,     #Multiply learning rate by 0.5
            verbose = True                 #Print when learning rate changes
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
            model.train() #Set model to training mode -> This enables dropout and batch norm updates

            trainingLoss = 0
            trainingCorrect = 0
            trainingTotal = 0

            progressBar = tqdm(trainLoder, desc = f"Epoch {epoch+1}/{epochs}")

            for images, labels in progressBar:
                #Move data to device (GPU if available)
                images = images.to(device)
                labels = labels.to(device)
                
                #Step 1: Zero gradients
                #WHY -> PyTorch accumulates gradients. Without this, gradients from last batch would add to current batch!
                optimizer.zero_grad()
                
                #Step 2: Forward pass
                #Feed images through network, get predictions
                outputs = model(images)  #Shape -> (batch_size, 2)
                
                #Step 3: Compute loss
                # Compare predictions to true labels
                loss = criterion(outputs, labels)
                
                #Step 4: Backward pass (backpropagation)
                #Compute gradient of loss with respect to every weight
                loss.backward()
                
                #Step 5: Update weights
                #newWeight = oldWeight - learningRate * gradient
                optimizer.step()
                
                trainingLoss += loss.item()
                
                #Get predictions (class with highest score)
                _, predicted = outputs.max(1)
                trainingTotal += labels.size(0)
                trainingCorrect += predicted.eq(labels).sum().item()
                
                #Update progress bar
                progressBar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{100.*trainingCorrect/trainingTotal:.2f}%"
                })

            #Calculating epoch averages
            trainingLoss = trainingLoss / len(trainLoder)
            trainingAcc = trainingCorrect / trainingTotal

            #Set model to evaluation mode
            #This disables dropout and uses fixed batch norm statistics
            model.eval()

            validationLoss = 0
            validationCorrect = 0
            validationTotal = 0
            
            #No gradient computation needed for validation
            #This saves memory and computation
            with torch.no_grad():
                for images, labels in validationLoader:
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    validationTotal += labels.size(0)
                    validationCorrect += predicted.eq(labels).sum().item()
            
            validationLoss = validationLoss / len(validationLoader)
            validationAcc = validationCorrect / validationTotal

            #Tell scheduler about validation loss
            scheduler.step(validationLoss)
            currentLearningRate = optimizer.param_groups[0]['lr']
            
            #Save if best model
            if validationAcc > bestValidationAccuracy:
                bestValidationAccuracy = validationAcc
                epochsWithoutImprovement = 0
                
                #Save model checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_acc': validationAcc,
                }, savePath)
                print(f'✅ New best model! Val Accuracy -> {validationAcc:.4f}')
            else:
                epochsWithoutImprovement += 1
            
            #Recording history
            history['trainLoss'].append(trainingLoss)
            history['trainAcc'].append(trainingAcc)
            history['validationLoss'].append(validationLoss)
            history['validationAcc'].append(validationAcc)
            history['learningRates'].append(currentLearningRate)
            
            #Printing epoch summary
            print(f'⏰ Epoch {epoch+1} -> Train Loss = {trainingLoss:.4f}, '
                f'Train Acc = {trainingAcc:.4f}, Val Acc = {validationAcc:.4f}')
            
            #Early stopping
            if epochsWithoutImprovement >= config.EARLY_STOPPING_PATIENCE:
                print(f'Early stopping -> no improvement for {epochsWithoutImprovement} epochs')
                break
        
        print(f'\n⏰ Training complete in {(time.time() - startTime)/60:.1f} minutes')
        print(f'🏹 Best validation accuracy -> {bestValidationAccuracy:.4f}')
        
        return history
    
    def loadCheckpoint(self, model, checkPointPath, device = None) -> None:
        if device is None:
            device = self.config.deviceSettings
            
        checkpoint = torch.load(checkPointPath, map_location = device)
        model.load_state_dict(checkpoint['model_state_dict'])
            
        print(f"⚜️ Loaded model from epoch {checkpoint['epoch']+1}")
        print(f"🏹Validation accuracy -> {checkpoint['val_acc']:.4f}")
            
        return model, checkpoint