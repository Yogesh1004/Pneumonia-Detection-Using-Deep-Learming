import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import time

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10 # Reduced for faster initial training

def get_actual_root(base_path):
    """Finds the directory containing 'train', 'test', 'val'."""
    actual_root = base_path
    for _ in range(3):
        if os.path.exists(os.path.join(actual_root, 'train')):
            return actual_root
        subdirs = [os.path.join(actual_root, d) for d in os.listdir(actual_root) if os.path.isdir(os.path.join(actual_root, d))]
        if subdirs:
            actual_root = subdirs[0]
        else:
            break
    return actual_root

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_path = "dataset"
    actual_root = get_actual_root(dataset_path)
    print(f"Using dataset root: {actual_root}")

    # Data Transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Data Loaders
    image_datasets = {x: datasets.ImageFolder(os.path.join(actual_root, x), data_transforms[x])
                      for x in ['train', 'val', 'test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True)
                   for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes
    print(f"Classes: {class_names}")

    # Build Model
    model = models.mobilenet_v2(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False # Freeze base model

    # Replace classifier
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 1),
        nn.Sigmoid()
    )

    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.classifier[1].parameters(), lr=0.001)

    # Training Loop
    best_acc = 0.0
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch}/{EPOCHS - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = (outputs > 0.5).float()
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'pneumonia_model.pth')

    print(f'Training complete. Best val Acc: {best_acc:4f}')

if __name__ == "__main__":
    train()
