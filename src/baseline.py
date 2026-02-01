import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

from . import config

class Baseline():
    def __init__(self):
        self.config = config.Config()
    
    def computeBaseline(self, trainDataset, testDataset, classNames, maxSamples = 5000) -> dict:
        print("\n" + "=" * 50)
        print("COMPUTING BASELINE")
        print("=" * 50)
        
        def extractFeatures(dataset, maxN):
            features = []
            labels = []
            
            n = min(maxN, len(dataset))
            indices = np.random.RandomState(self.config.reproducibilitySeed).permutation(len(dataset))[:n]
            
            print(f"🗄️ Extracting features from {n} images...")
            
            for idx in tqdm(indices):
                img, label = dataset[idx]
                features.append(img.numpy().flatten())
                labels.append(label)
            
            return np.array(features), np.array(labels)
        
        xTrain, yTrain = extractFeatures(trainDataset, maxSamples)
        xTest, yTest = extractFeatures(testDataset, maxSamples // 2)
        
        print(f"♨️ Feature vector size: {xTrain.shape[1]}")
        
        scaler = StandardScaler()
        xTrain = scaler.fit_transform(xTrain)
        xTest = scaler.transform(xTest)  # FIXED: was X_test
        
        print("🔄 Training Logistic Regression...")
        model = LogisticRegression(max_iter = 1000, random_state = self.config.reproducibilitySeed)
        model.fit(xTrain, yTrain)
        
        predictions = model.predict(xTest)
        accuracy = accuracy_score(yTest, predictions)
        f1 = f1_score(yTest, predictions)
        
        print(f"\n📊 Baseline Results:")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  F1 Score: {f1:.4f}")
        print("\n" + classification_report(yTest, predictions, target_names = classNames))
        
        return {'accuracy': accuracy, 'f1': f1}