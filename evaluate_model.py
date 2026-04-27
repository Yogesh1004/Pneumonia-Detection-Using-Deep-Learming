import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

def get_actual_root(base_path):
    """Finds the directory containing 'train', 'test', 'val'."""
    actual_root = base_path
    for _ in range(3):
        if os.path.exists(os.path.join(actual_root, 'test')):
            return actual_root
        subdirs = [os.path.join(actual_root, d) for d in os.listdir(actual_root) if os.path.isdir(os.path.join(actual_root, d))]
        if subdirs:
            actual_root = subdirs[0]
        else:
            break
    return actual_root

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")

    dataset_path = "dataset"
    actual_root = get_actual_root(dataset_path)
    test_dir = os.path.join(actual_root, 'test')

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    dataset_size = len(test_dataset)

    # Rebuild model architecture
    model = models.mobilenet_v2(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 1),
        nn.Sigmoid()
    )

    # Load weights
    model.load_state_dict(torch.load('pneumonia_model.pth', map_location=device))
    model = model.to(device)
    model.eval()

    running_corrects = 0
    
    print(f"Starting evaluation on {dataset_size} images...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(inputs)
            preds = (outputs > 0.5).float()
            running_corrects += torch.sum(preds == labels.data).item()

    accuracy = running_corrects / dataset_size
    print(f"Final Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    evaluate()
