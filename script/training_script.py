import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

torch.cuda.empty_cache()
# Initialize device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data augmentation and normalization for training
# More aggressive augmentation for 'ok'
data_transforms = {
    'ok': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'ng': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Create datasets and dataloaders
ok_data_dir = 'ok'
ng_data_dir = 'ng'  # Changed this line

image_datasets = {
    'ok': datasets.ImageFolder(ok_data_dir, data_transforms['ok']),
    'ng': datasets.ImageFolder(ng_data_dir, data_transforms['ng'])  # And this line
}

dataloaders = {x: DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=1) for x in ['ok', 'ng']}


# Create model
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features

# Number of classes should match the number of folders inside 'ok'
num_classes = len(os.listdir(ok_data_dir))  # Automatically count the number of subdirectories
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

# Loss and optimizer with increased learning rate
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # Increased learning rate

# Training loop
num_epochs = 30
for epoch in range(num_epochs):
    for phase in ['ok', 'ng']:
        if phase == 'ok':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'ok'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'ok':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(image_datasets[phase])
        print(f"{phase} Loss: {epoch_loss:.4f}")

# Save trained model
torch.save(model.state_dict(), 'save_model.pth')

print("Training complete.")
