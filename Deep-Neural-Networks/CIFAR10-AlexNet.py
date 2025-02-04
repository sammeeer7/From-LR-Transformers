import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import numpy as np

class AlexNetCIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNetCIFAR10, self).__init__()
        
        self.features = nn.Sequential(
            # Layer 1 (Conv1)
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Layer 2 (Conv2)
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Layer 3 (Conv3)
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Layer 4 (Conv4)
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Layer 5 (Conv5)
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self._to_linear = None
        self._calculate_fc_input(torch.zeros(1, 3, 32, 32))
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self._to_linear, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
        
        self._initialize_weights()
    
    def _calculate_fc_input(self, x):
        with torch.no_grad():
            x = self.features(x)
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def get_data_loaders():
    # Data augmentation and normalization for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    ])
    
    # Only normalization for validation
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    ])
    
    # Load CIFAR-10 datasets
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    valset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_val)
    
    # Create data loaders
    trainloader = DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    valloader = DataLoader(
        valset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    
    return trainloader, valloader

def train_model(model, trainloader, valloader, num_epochs=90):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    
    # Training loop
    best_acc = 0.0
    training_stats = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        start_time = time.time()
        
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': running_loss/total,
                'acc': 100.*correct/total
            })
        
        train_loss = running_loss/total
        train_acc = 100.*correct/total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss/len(valloader)
        val_acc = 100.*correct/total
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, 'alexnet_cifar10_best.pth')
        
        # Print statistics
        print(f'\nEpoch {epoch+1}:')
        print(f'Training Loss: {train_loss:.3f} | Training Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.2f}%')
        print(f'Time: {time.time() - start_time:.2f}s')
        
        training_stats.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
        })
    
    return model, training_stats

def main():
    # Print GPU info
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("No GPU available, using CPU")
    
    # Create model
    model = AlexNetCIFAR10(num_classes=10)
    
    # Get data loaders
    trainloader, valloader = get_data_loaders()
    
    # Train model
    print("Starting training...")
    model, stats = train_model(model, trainloader, valloader)
    
    # Print final results
    print("\nTraining completed!")
    print(f"Best validation accuracy: {max(stat['val_acc'] for stat in stats):.2f}%")

if __name__ == "__main__":
    main()