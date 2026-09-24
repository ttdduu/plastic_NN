import torch
import numpy as np

class Metric:
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        pass

    def update(self, outputs, targets):
        pass

    def compute(self):
        pass

class Accuracy(Metric):
    def __init__(self):
        super().__init__('accuracy')
        self.correct = 0
        self.total = 0

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, outputs, targets):
        _, predicted = torch.max(outputs, 1)
        self.correct += (predicted == targets).sum().item()
        self.total += targets.size(0)

    def compute(self):
        return self.correct / self.total if self.total > 0 else 0

class Loss(Metric):
    def __init__(self):
        super().__init__('loss')
        self.sum = 0
        self.count = 0

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, loss, batch_size):
        self.sum += loss * batch_size
        self.count += batch_size

    def compute(self):
        return self.sum / self.count if self.count > 0 else 0

def compute_confusion_matrix(model, data_loader, config, num_classes):
    model.eval()
    confusion_matrix = np.zeros((num_classes, num_classes))
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(config.training.device)
            targets = targets.to(config.training.device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            for t, p in zip(targets.cpu().numpy(), predicted.cpu().numpy()):
                confusion_matrix[t, p] += 1
                
    # Normalize the confusion matrix
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    normalized_cm = confusion_matrix / row_sums
    
    return normalized_cm

# Add more metrics as needed
