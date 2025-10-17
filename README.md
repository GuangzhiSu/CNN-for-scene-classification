# CNN for Scene Classification

This repository contains the team project for **Duke AIPI 590 Applied Computer Vision** class. The project focuses on implementing and comparing different CNN architectures for scene classification using the SUN397 dataset.

## Team Members

- **Guangzhi Su** - ResNet-50 fine-tuning, dropout comparison, activation function analysis
- **Ryan** - Comprehensive experiments and model optimization
- **Shuaiming** - Data augmentation techniques and model training

## Project Overview

This project implements and evaluates various CNN architectures for scene classification, with a focus on:

- **Transfer Learning**: Fine-tuning pre-trained models (ResNet-50) for scene classification
- **Regularization Techniques**: Comparing dropout effectiveness in preventing overfitting
- **Activation Functions**: Analyzing the impact of different activation functions (ELU, ReLU, SiLU)
- **Data Augmentation**: Exploring various augmentation strategies to improve model generalization
- **Model Optimization**: Comprehensive hyperparameter tuning and architecture modifications

## Dataset

- **SUN397 Dataset**: Large-scale scene recognition dataset with 397 scene categories
- **Data Split**: 70% training, 10% validation, 20% testing
- **Image Preprocessing**: Standard ImageNet normalization and augmentation techniques

## Project Structure

```
CNN-for-scene-classification/
├── README.md
├── Guangzhi Su/
│   ├── ResNet-50-finetune.py          # Main ResNet-50 fine-tuning implementation
│   ├── Dropout_Comparison.py          # Dropout effectiveness analysis
│   ├── activation_comparison.py       # Activation function comparison
│   ├── confusion_matrix.png           # Model performance visualization
│   ├── dropout_comparison.png         # Dropout analysis results
│   └── activation_comparison.png      # Activation function results
├── Ryan/
│   ├── CVTeamProject_1_10Experiments-OutputsComplete.ipynb
│   ├── CVTeamProject1_BestFullDataset101.ipynb
│   ├── CVTeamProject1_BestFullDataset18.ipynb
│   └── CVTeamProject1_FineTuningTests_OutputsComplete.ipynb
└── Shuaiming/
    ├── CV_project1_augfull.ipynb
    ├── CV_project1_augpartipynb.ipynb
    └── CV_project1.ipynb
```

## Key Contributions

### Guangzhi Su's Work
- **ResNet-50 Fine-tuning**: Implemented transfer learning with frozen feature extractor and custom classifier
- **Dropout Analysis**: Comparative study of models with and without dropout regularization
- **Activation Function Study**: Evaluation of ELU vs ReLU activation functions
- **Performance Visualization**: Comprehensive confusion matrices and loss curves

### Ryan's Work
- **Comprehensive Experiments**: Extensive hyperparameter tuning and model optimization
- **Full Dataset Training**: Training on complete SUN397 dataset with 101 and 18 class subsets
- **Fine-tuning Tests**: Detailed analysis of different fine-tuning strategies

### Shuaiming's Work
- **Data Augmentation**: Implementation and evaluation of various augmentation techniques
- **Partial Dataset Training**: Training strategies for smaller dataset subsets
- **Model Training**: End-to-end training pipeline implementation

## Technical Implementation

### Model Architecture
- **Base Model**: ResNet-50 pre-trained on ImageNet
- **Feature Extractor**: Frozen pre-trained layers for transfer learning
- **Classifier**: Custom fully connected layers with configurable activation functions
- **Regularization**: Dropout layers for overfitting prevention

### Training Configuration
- **Optimizer**: SGD with Nesterov momentum (lr=0.01, momentum=0.9)
- **Scheduler**: StepLR with step_size=3-5, gamma=0.7
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 32
- **Epochs**: 10-20 depending on experiment

### Data Augmentation
- **Training**: RandomResizedCrop(224), RandomHorizontalFlip
- **Validation/Test**: Resize(256), CenterCrop(224)
- **Normalization**: ImageNet mean and std values

## Results and Analysis

### Key Findings
1. **Transfer Learning Effectiveness**: Pre-trained ResNet-50 features significantly improve scene classification performance
2. **Dropout Impact**: Dropout regularization helps prevent overfitting, especially with limited data
3. **Activation Functions**: ELU shows competitive performance compared to ReLU in scene classification tasks
4. **Data Augmentation**: Proper augmentation strategies improve model generalization

### Performance Metrics
- **Accuracy**: Achieved competitive accuracy on SUN397 scene classification
- **Confusion Matrix**: Detailed per-class performance analysis
- **Loss Curves**: Training and validation loss tracking for model convergence analysis

## Usage

### Prerequisites
```bash
pip install torch torchvision matplotlib seaborn scikit-learn tqdm pillow
```

### Running Experiments

1. **ResNet-50 Fine-tuning**:
```bash
cd "Guangzhi Su"
python ResNet-50-finetune.py
```

2. **Dropout Comparison**:
```bash
cd "Guangzhi Su"
python Dropout_Comparison.py
```

3. **Activation Function Analysis**:
```bash
cd "Guangzhi Su"
python activation_comparison.py
```

### Jupyter Notebooks
- Navigate to respective team member folders
- Open and run the Jupyter notebooks for comprehensive experiments

## Key Features

- **Modular Design**: Separate scripts for different experiments
- **Comprehensive Logging**: Detailed training progress and metrics
- **Visualization**: Automatic generation of confusion matrices and loss curves
- **GPU Support**: CUDA acceleration for faster training
- **Progress Tracking**: tqdm progress bars for training monitoring

## Future Work

- [ ] Implement additional CNN architectures (VGG, DenseNet, EfficientNet)
- [ ] Explore advanced data augmentation techniques (Mixup, CutMix)
- [ ] Investigate attention mechanisms for scene classification
- [ ] Optimize hyperparameters using automated tuning
- [ ] Implement ensemble methods for improved performance

## Acknowledgments

- Duke University AIPI 590 Applied Computer Vision Course
- SUN397 Dataset creators
- PyTorch and torchvision development teams
- Team collaboration and knowledge sharing

## License

This project is developed for educational purposes as part of Duke University's AIPI 590 course.
