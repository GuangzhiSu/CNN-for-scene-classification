import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.optim.lr_scheduler
import os
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# Specify the file path
data_dir = "/home/gs285/HW/AIPI/Proj_1/SUN397/a"

save_dir = '/home/gs285/HW/AIPI/Proj_1/codeOnA'
print("Starting dataset loading...")




# Define transformations
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define transformations for the validation and test sets (without augmentation)
val_test_transform = transforms.Compose([
    transforms.Resize(256),  # Resize to a fixed size
    transforms.CenterCrop(224),  # Crop the center
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

full_dataset = datasets.ImageFolder(data_dir)
class_names = full_dataset.classes
print(f"Class names: {class_names}")

# Define split ratios
train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2

# Calculate the sizes for each split
data_len = len(full_dataset)
train_size = int(train_ratio * data_len)
val_size = int(val_ratio * data_len)
test_size = data_len - train_size - val_size

# Perform the split
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])


# Apply transformations to the datasets
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_test_transform
test_dataset.dataset.transform = val_test_transform


# Define the data loaders for training, validation, and testing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# Load the VGG16 model pretrained on ImageNet
resnet = models.resnet50(pretrained=True)

# Freeze all layers in the pre-trained network
for param in resnet.parameters():
    param.requires_grad = False

print("Model loaded successfully.")

# Modify the classifier to have fewer parameters and match the number of scene classes
class SceneRecognitionModel(nn.Module):
    def __init__(self, num_classes=20):
        super(SceneRecognitionModel, self).__init__()
        # Use the pre-trained ResNet as the feature extractor
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # Remove the final classification layer
        
        self.classifier = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 1024),  # Adjust for ResNet's output size
            nn.SiLU(),  # Using Swish
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        # Forward pass through the frozen ResNet feature extractor
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the feature maps
        # Forward pass through the new classifier
        x = self.classifier(x)
        return x

# Create the model
model = SceneRecognitionModel(num_classes = 20)

print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Model moved to {device}")

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

total_params = count_trainable_params(model)
print(f"Total trainable parameters: {total_params}")

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.classifier.parameters(), 
    lr=0.01,  # Initial learning rate (adjust this as needed)
    momentum=0.9,  # Momentum factor
    nesterov=True  # Enable Nesterov momentum
)

# Define a learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)  # Decays the learning rate every 5 epochs by a factor of 0.1


# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, save_dir, num_epochs=10):
    model.train()
    epoch_losses = []
    val_accuracies = []
    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        running_loss = 0.0

        # Initialize progress bar for each epoch
        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{num_epochs}")

            for images, labels in tepoch:
                # Move data to device (CPU or GPU)
                images, labels = images.to(device), labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Update progress bar
                tepoch.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader)
        epoch_losses.append(epoch_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss:.4f}')
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            with tqdm(val_loader, unit="batch") as vepoch:
                vepoch.set_description(f"Validating Epoch {epoch + 1}/{num_epochs}")
                
                for val_images, val_labels in vepoch:
                    val_images, val_labels = val_images.to(device), val_labels.to(device)

                    val_outputs = model(val_images)
                    val_loss += criterion(val_outputs, val_labels).item()
                    _, val_preds = torch.max(val_outputs, 1)
                    correct += (val_preds == val_labels).sum().item()
                    total += val_labels.size(0)

        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)
        val_loss /= len(val_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        # Step the scheduler after each epoch
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch + 1}/{num_epochs}], Learning Rate: {current_lr:.6f}")

    # Save model after each epoch
    weight_save_path = os.path.join(save_dir, f'model_weights_epoch_{epoch+1}.pth')
    torch.save(model.state_dict(), weight_save_path)
    print(f"Model weights saved to {weight_save_path}")

    return epoch_losses, val_accuracies


def plot_confusion_matrix(cm, class_names, save_path, normalize=True):
    plt.figure(figsize=(10, 8))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize by row (true class)

    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()



# Evaluation function
def evaluate_model(model, test_loader, class_names, save_dir):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    print("Starting evaluation...")

    with tqdm(test_loader, unit="batch") as ttest:
        ttest.set_description(f"Evaluating Model")
        with torch.no_grad():
            for images, labels in ttest:
                # Move images and labels to the appropriate device
                images, labels = images.to(device), labels.to(device)
                # Forward pass through the model
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                # Collect predictions and labels
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    # Calculate final accuracy
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on test images: {accuracy:.2f}%')

    # Get confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Save and plot confusion matrix
    cm_save_path = os.path.join(save_dir, 'confusion_matrix.png')
    plot_confusion_matrix(cm, class_names, cm_save_path, normalize=True)

    # Return accuracy and confusion matrix for further analysis
    return accuracy, cm


# Number of epochs for training
num_epochs = 10



if __name__ == "__main__":

    # Training the model if no saved weights are found or train_mode is set to True
    print("Starting training...")
    epoch_losses, val_accuracies = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, save_dir, num_epochs=num_epochs)

    # After either loading the model or training, evaluate on the test set
    print("Starting evaluation on test set...")
    test_accuracy, cm = evaluate_model(model, test_loader, class_names, save_dir)
    print(f"Final accuracy on test set: {test_accuracy:.2f}%")
