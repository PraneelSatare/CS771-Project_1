# Multi-Modal Binary Classification with Ensemble Learning
**CS771 Mini-Project 1 - Team t-AI-tans**

## Team Members
- **Akshat Sharma** (230101)
- **Dweep Joshipura** (230395)
- **Kanak Khandelwal** (230520)
- **Praneel B Satare** (230774) 

## Project Overview

This project implements a comprehensive machine learning pipeline for binary classification across three distinct feature representations of the same dataset. We explore classical ML approaches, deep learning techniques, and ensemble methods to achieve optimal performance across different data modalities.

### Key Contributions
- **Multi-modal Analysis**: Comparative study across emoticon, deep features, and text sequence datasets
- **Comprehensive Model Evaluation**: Testing 15+ different algorithms including classical ML and neural networks
- **Advanced Preprocessing**: Feature engineering with PCA, standardization, and custom binary matrix encoding
- **Ensemble Learning**: Weighted voting approach for combining predictions from different modalities

## Dataset Description

The project works with three feature representations of the same underlying dataset:

### 1. Emoticon Dataset
- **Format**: Strings of emojis (max length 13)
- **Features**: 214 unique emojis converted to binary matrices (214 × 13)
- **Preprocessing**: Binary matrix encoding with position-based representation
- **Challenge**: Sparse, high-dimensional feature space

### 2. Deep Features Dataset  
- **Format**: Pre-extracted deep features (13 × 786 matrices)
- **Features**: 10,218-dimensional flattened vectors
- **Preprocessing**: PCA for dimensionality reduction, StandardScaler normalization
- **Challenge**: High-dimensional continuous features

### 3. Text Sequence Dataset
- **Format**: Digit sequences of length 50
- **Features**: Sequential numerical data (0-9)
- **Preprocessing**: Column-wise standardization (μ=0, σ=1)
- **Challenge**: Temporal dependencies and sequential patterns

## Model Architectures & Results

### Task 1a: Emoticon Classification
**Best Model**: Logistic Regression with L1 Regularization
- **Validation Accuracy**: 93.25%
- **Key Features**: 
  - Effective handling of sparse binary features
  - Automatic feature selection through L1 penalty
  - Optimal regularization strength: C = 0.2

| Model | 20% | 40% | 60% | 80% | 100% |
|-------|-----|-----|-----|-----|------|
| Logistic Regression (L1) | 74.85% | 82.41% | 85.48% | 89.16% | **93.25%** |
| Perceptron | 73.42% | 81.60% | 84.46% | 86.09% | 91.21% |
| SVM (Linear) | 72.60% | 80.37% | 83.44% | 85.48% | 88.34% |

### Task 1b: Deep Features Classification
**Best Model**: SVM with RBF Kernel
- **Validation Accuracy**: 99.18%
- **Key Features**:
  - Superior performance in high-dimensional space
  - Effective kernel trick for non-linear boundaries
  - Hyperparameter tuning with grid search

| Model | 20% | 40% | 60% | 80% | 100% |
|-------|-----|-----|-----|-----|------|
| SVM (RBF Kernel) | 95.09% | 97.55% | 97.96% | **99.18%** | 98.77% |
| Logistic Regression (L1) | 96.32% | 98.16% | 97.75% | 98.36% | 98.16% |
| Decision Tree | 90.59% | 92.84% | 94.68% | 95.91% | 94.07% |

### Task 1c: Text Sequence Classification
**Best Model**: LSTM Neural Network
- **Validation Accuracy**: 75.87%
- **Architecture**: 
  - LSTM layer (32 units) + Dense layer (1 unit, sigmoid)
  - Adam optimizer (lr=0.01, batch_size=128)
  - Total parameters: 4,385

| Model | 20% | 40% | 60% | 80% | 100% |
|-------|-----|-----|-----|-----|------|
| LSTM | 59.71% | 63.19% | 68.30% | 69.12% | **75.87%** |
| Random Forest | 58.49% | 61.15% | 60.74% | 60.53% | 63.80% |
| Fully Convolutional Network | 52.56% | 56.03% | 51.74% | 69.12% | 61.55% |

### Task 2: Ensemble Learning
**Combined Approach**: Weighted Voting Ensemble
- **Final Accuracy**: 99.2% (equivalent to best individual model)
- **Method**: SLSQP optimization for learning optimal weights
- **Learned Weights**: [0.15, 0.7, 0.15] (heavily favoring SVM predictions)

## Technical Implementation

### Key Algorithms Implemented
- **Classical ML**: Logistic Regression, SVM (Linear/RBF/Polynomial), Random Forest, Naive Bayes
- **Deep Learning**: LSTM, Fully Convolutional Networks, Multi-layer Perceptron
- **Preprocessing**: PCA, StandardScaler, Custom binary encoding
- **Ensemble**: Majority voting, Weighted voting with learned parameters

### Feature Engineering Highlights
1. **Emoticon Encoding**: Novel binary matrix representation preserving positional information
2. **Dimensionality Reduction**: PCA analysis revealing data separability
3. **Sequential Processing**: Column-wise standardization for temporal data
4. **Ensemble Weighting**: Optimization-based weight learning for model combination

## Usage Instructions

### Prerequisites
```bash
pip install numpy pandas scikit-learn joblib keras tensorflow
```

### Setup Instructions
1. **Extract the SVM model**:
   ```bash
   # Extract the compressed SVM model file
   unzip svm_max.zip
   ```
   This will create `svm_max.pkl` required by the main script.

2. **Run the Pipeline**:
   ```bash
   python group_7.py
   ```

The script will prompt for dataset paths:
1. Training dataset path for emoticon dataset
2. Testing dataset path for emoticon dataset  
3. Testing dataset path for deep features dataset
4. Testing dataset path for text sequences dataset

## Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/djthegr8/CS771-Project-1
cd CS771-Project-1

# Install dependencies
pip install numpy pandas scikit-learn joblib keras tensorflow

# Extract the compressed SVM model
unzip svm_max.zip
```

### 2. Run the Pipeline
```bash
python group_7.py
```

### 3. Provide Dataset Paths
When prompted, provide the full paths to your datasets:
- Training emoticon dataset (CSV format)
- Testing emoticon dataset (CSV format)  
- Testing deep features dataset (NPZ format)
- Testing text sequences dataset (CSV format)

### 4. Check Results
The script will generate four prediction files:
- `pred_emoticon.txt` - Emoticon dataset predictions
- `pred_deepfeat.txt` - Deep features dataset predictions  
- `pred_text.txt` - Text sequence dataset predictions
- `pred_combined.txt` - Combined ensemble predictions

## Project Structure
```
├── group_7.py                                      # Main execution script
├── read_data.py                                    # Data loading utilities  
├── read_data_mod.py                               # Modified data loading with user input
├── report.pdf                                     # Detailed technical report
├── README.md                                      # Project documentation
├── lstm_trained_on_100_percent_lr=0.01_bs=128.keras  # Pre-trained LSTM model
├── scaler.joblib                                  # Feature scaler for LSTM
├── scaler.pkl                                     # Feature scaler for SVM
├── svm_max.zip                                    # Compressed SVM model (needs extraction)
└── .git/                                          # Git repository folder
```

### Important Notes
- **Model Files**: The SVM model (`svm_max.pkl`) is compressed as `svm_max.zip` and must be extracted before running the code
- **Pre-trained Models**: LSTM and SVM models are pre-trained and loaded during execution
- **Scalers**: Separate scalers are used for different model pipelines

## Key Insights & Findings

### Model Selection Rationale
1. **Sparse Data (Emoticons)**: L1 regularization effectively handles high-dimensional sparse binary features
2. **Dense Features**: SVM with RBF kernel excels in high-dimensional continuous spaces
3. **Sequential Data**: LSTM captures temporal dependencies that classical models miss
4. **Ensemble Limitation**: When one model significantly outperforms others (99%+ accuracy), ensemble benefits are minimal

### Performance Analysis
- **Feature Engineering Impact**: Proper preprocessing improved accuracy by ~3% across models
- **Data Scaling Effects**: Sequential patterns emerge more clearly with standardized features
- **Regularization Benefits**: L1 penalty provides automatic feature selection in sparse domains

## Future Improvements
1. **Probabilistic Ensembles**: Use confidence scores instead of hard predictions
2. **Advanced Architectures**: Transformer models for sequential data
3. **Cross-modal Learning**: Joint training across all three modalities
4. **Hyperparameter Optimization**: Automated tuning with Bayesian optimization

## References & Acknowledgments
This project was completed as part of CS771 (Introduction to Machine Learning) course. The comprehensive analysis demonstrates the importance of choosing appropriate models for different data characteristics and the potential of ensemble methods in multi-modal scenarios.

---
*For detailed mathematical formulations, experimental results, and additional analysis, please refer to the included technical report (report.pdf).*
