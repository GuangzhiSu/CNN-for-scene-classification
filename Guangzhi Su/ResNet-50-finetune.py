import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.optim.lr_scheduler
import os
import yaml
import json
import logging
from datetime import datetime
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
import numpy as np
import argparse

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(save_dir, level='INFO'):
    """Setup logging configuration"""
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Load configuration
config = load_config()
logger = setup_logging(config['paths']['save_dir'], config['logging']['level'])
set_seed()

# Extract configuration values
data_dir = config['data']['data_dir']
save_dir = config['paths']['save_dir']
batch_size = config['data']['batch_size']
num_workers = config['data']['num_workers']
num_classes = config['model']['num_classes']
num_epochs = config['training']['num_epochs']

logger.info("Starting dataset loading...")
logger.info(f"Configuration loaded: {config}")




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

try:
    full_dataset = datasets.ImageFolder(data_dir)
    class_names = full_dataset.classes
    logger.info(f"Class names: {class_names}")
    logger.info(f"Total number of classes: {len(class_names)}")
    
    # Define split ratios from config
    train_ratio = config['data']['train_ratio']
    val_ratio = config['data']['val_ratio']
    test_ratio = config['data']['test_ratio']
    
    # Calculate the sizes for each split
    data_len = len(full_dataset)
    train_size = int(train_ratio * data_len)
    val_size = int(val_ratio * data_len)
    test_size = data_len - train_size - val_size
    
    logger.info(f"Dataset split - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Perform the split
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    
    # Apply transformations to the datasets
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_test_transform
    test_dataset.dataset.transform = val_test_transform
    
    # Define the data loaders for training, validation, and testing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
except Exception as e:
    logger.error(f"Error loading dataset: {e}")
    raise

# Load the ResNet-50 model pretrained on ImageNet
try:
    resnet = models.resnet50(pretrained=config['model']['pretrained'])
    logger.info("ResNet-50 model loaded successfully.")
    
    # Freeze all layers in the pre-trained network if specified
    if config['model']['freeze_backbone']:
        for param in resnet.parameters():
            param.requires_grad = False
        logger.info("Backbone layers frozen.")
    
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

# Modify the classifier to have fewer parameters and match the number of scene classes
class SceneRecognitionModel(nn.Module):
    def __init__(self, num_classes=20, hidden_size=1024, dropout_rate=0.5):
        super(SceneRecognitionModel, self).__init__()
        # Use the pre-trained ResNet as the feature extractor
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # Remove the final classification layer
        
        self.classifier = nn.Sequential(
            nn.Linear(resnet.fc.in_features, hidden_size),  # Adjust for ResNet's output size
            nn.SiLU(),  # Using Swish activation
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        # Forward pass through the frozen ResNet feature extractor
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the feature maps
        # Forward pass through the new classifier
        x = self.classifier(x)
        return x

# Create the model
model = SceneRecognitionModel(
    num_classes=num_classes,
    hidden_size=config['model']['classifier_hidden_size'],
    dropout_rate=config['model']['dropout_rate']
)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
logger.info(f"Model moved to {device}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

total_params = count_trainable_params(model)
logger.info(f"Total trainable parameters: {total_params:,}")

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.classifier.parameters(), 
    lr=config['training']['learning_rate'],
    momentum=config['training']['momentum'],
    weight_decay=config['training']['weight_decay'],
    nesterov=config['training']['nesterov']
)

# Define a learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, 
    step_size=config['training']['scheduler']['step_size'], 
    gamma=config['training']['scheduler']['gamma']
)

logger.info(f"Optimizer: {optimizer}")
logger.info(f"Scheduler: {scheduler}")

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_model_state = None
    
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict().copy()
        else:
            self.counter += 1
        
        return self.counter >= self.patience

# Initialize early stopping if enabled
early_stopping = None
if config['early_stopping']['enabled']:
    early_stopping = EarlyStopping(
        patience=config['early_stopping']['patience'],
        min_delta=config['early_stopping']['min_delta']
    )

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, save_dir, num_epochs=10):
    model.train()
    epoch_losses = []
    val_accuracies = []
    val_losses = []
    best_val_accuracy = 0.0
    
    logger.info(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        try:
            running_loss = 0.0
            model.train()

            # Initialize progress bar for each epoch
            with tqdm(train_loader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}/{num_epochs}")

                for batch_idx, (images, labels) in enumerate(tepoch):
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
            logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss:.4f}')
            
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
            val_losses.append(val_loss)
            
            logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

            # Step the scheduler after each epoch
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Learning Rate: {current_lr:.6f}")

            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_path = os.path.join(save_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_accuracy': val_accuracy,
                    'val_loss': val_loss,
                    'class_names': class_names,
                    'config': config
                }, best_model_path)
                logger.info(f"New best model saved with validation accuracy: {val_accuracy:.2f}%")

            # Save model after each epoch
            weight_save_path = os.path.join(save_dir, f'model_weights_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_accuracy': val_accuracy,
                'val_loss': val_loss,
                'class_names': class_names,
                'config': config
            }, weight_save_path)
            logger.info(f"Model weights saved to {weight_save_path}")

            # Early stopping check
            if early_stopping and early_stopping(val_loss, model):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                if early_stopping.best_model_state:
                    model.load_state_dict(early_stopping.best_model_state)
                    logger.info("Restored best model state")
                break

        except Exception as e:
            logger.error(f"Error during epoch {epoch + 1}: {e}")
            # Save current progress
            emergency_save_path = os.path.join(save_dir, f'emergency_save_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), emergency_save_path)
            logger.info(f"Emergency save created at {emergency_save_path}")
            raise

    # Save training history
    training_history = {
        'epoch_losses': epoch_losses,
        'val_accuracies': val_accuracies,
        'val_losses': val_losses,
        'best_val_accuracy': best_val_accuracy,
        'config': config
    }
    
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    logger.info(f"Training history saved to {history_path}")

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



# Comprehensive evaluation function
def evaluate_model(model, test_loader, class_names, save_dir):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    logger.info("Starting comprehensive evaluation...")

    with tqdm(test_loader, unit="batch") as ttest:
        ttest.set_description(f"Evaluating Model")
        with torch.no_grad():
            for images, labels in ttest:
                # Move images and labels to the appropriate device
                images, labels = images.to(device), labels.to(device)
                # Forward pass through the model
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)

                # Collect predictions, labels, and probabilities
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    # Calculate final accuracy
    accuracy = 100 * correct / total
    logger.info(f'Accuracy of the model on test images: {accuracy:.2f}%')

    # Calculate additional metrics
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    logger.info(f'F1 Score (Macro): {f1_macro:.4f}')
    logger.info(f'F1 Score (Weighted): {f1_weighted:.4f}')

    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    logger.info(f"Classification Report:\n{report}")

    # Get confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Save and plot confusion matrix
    cm_save_path = os.path.join(save_dir, 'confusion_matrix.png')
    plot_confusion_matrix(cm, class_names, cm_save_path, normalize=True)

    # Save detailed evaluation results
    evaluation_results = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'predictions': all_preds,
        'true_labels': all_labels,
        'probabilities': all_probs
    }
    
    results_path = os.path.join(save_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    logger.info(f"Detailed evaluation results saved to {results_path}")

    # Return comprehensive results
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm,
        'classification_report': report
    }


if __name__ == "__main__":
    try:
        # Training the model
        logger.info("Starting training...")
        epoch_losses, val_accuracies = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler, save_dir, num_epochs=num_epochs
        )

        # After training, evaluate on the test set
        logger.info("Starting evaluation on test set...")
        evaluation_results = evaluate_model(model, test_loader, class_names, save_dir)
        
        logger.info(f"Final accuracy on test set: {evaluation_results['accuracy']:.2f}%")
        logger.info(f"Final F1 Score (Macro): {evaluation_results['f1_macro']:.4f}")
        logger.info(f"Final F1 Score (Weighted): {evaluation_results['f1_weighted']:.4f}")
        
        # Create summary report
        summary = {
            'final_test_accuracy': evaluation_results['accuracy'],
            'final_f1_macro': evaluation_results['f1_macro'],
            'final_f1_weighted': evaluation_results['f1_weighted'],
            'best_val_accuracy': max(val_accuracies) if val_accuracies else 0,
            'total_epochs': len(epoch_losses),
            'config_used': config,
            'timestamp': datetime.now().isoformat()
        }
        
        summary_path = os.path.join(save_dir, 'experiment_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Experiment summary saved to {summary_path}")
        
        logger.info("Training and evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
