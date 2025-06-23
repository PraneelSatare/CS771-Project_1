import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import pickle
from keras.models import load_model
from joblib import load
import read_data_mod

# --------------------------------
# Load file paths for different datasets
# --------------------------------
train_path_emoticon, test_path_emoticon, test_path_deep, test_path_text = read_data_mod.get_dataset_paths()

# --------------------------------
#   Emoticons Dataset
# --------------------------------

# ------------------------
# 1. Data Preparation
# ------------------------

# Load the training and test datasets
train_df = pd.read_csv(train_path_emoticon)
test_df = pd.read_csv(test_path_emoticon)

# Get the unique set of emojis from the training dataset and create a fixed sequence
unique_emojis = set(''.join(train_df['input_emoticon'].values))
fixed_emoji_sequence = sorted(unique_emojis)

# Function to convert emoji string to a binary matrix
def emoji_to_binary_matrix(emoji_string, fixed_sequence):
    n = len(fixed_sequence)
    matrix = np.zeros((n, 13), dtype=int)  # n x 13 matrix initialized to 0

    for j, emoji in enumerate(emoji_string):
        if emoji in fixed_sequence:
            i = fixed_sequence.index(emoji)
            matrix[i, j] = 1
    return matrix

# Transform the input_emoticon column to binary matrices for both train and test datasets
X_train = np.array([emoji_to_binary_matrix(emoticons, fixed_emoji_sequence) for emoticons in train_df['input_emoticon']])
X_test = np.array([emoji_to_binary_matrix(emoticons, fixed_emoji_sequence) for emoticons in test_df['input_emoticon']])

# Reshape X to flatten the features for Logistic Regression input
X_train_flat = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
X_test_flat = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2]))

# Extract labels for training
y_train = train_df['label'].values

# ------------------------
# 2. Train the Model
# ------------------------

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_test_scaled = scaler.transform(X_test_flat)  # Use the same scaler for the test set

# Initialize and configure Logistic Regression model with L1 regularization
log_reg_l1 = LogisticRegression(
    penalty='l1',
    solver='liblinear',  # Solver that supports L1 penalty
    max_iter=1000,
    random_state=42,
    C=0.2  # Regularization strength (can be tuned)
)

# Train the Logistic Regression model
log_reg_l1.fit(X_train_scaled, y_train)

# ------------------------
# 3. Make Predictions on Test Dataset
# ------------------------

# Make predictions on the test set
y_pred_test = log_reg_l1.predict(X_test_scaled)

# ------------------------
# 4. Save Predictions to File
# ------------------------

# Save the predictions to a file
output_file = 'pred_emoticon.txt'
with open(output_file, 'w') as f:
    for prediction in y_pred_test:
        f.write(f"{prediction}\n")

# --------------------------------
#   Deep Features Dataset
# --------------------------------

# ------------------------
# 1. Load and Preprocess the Test Data
# ------------------------

DATA = np.load(test_path_deep, allow_pickle=True)
x_test = DATA['features']
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

# ------------------------
# 2. Load the Saved SVM Model and Make Predictions
# ------------------------

with open('svm_max.pkl', 'rb') as f:
    svm_model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
    
# Scale the test data and make predictions
y_pred_svm = svm_model.predict(scaler.transform(x_test))

# ------------------------
# 3. Save SVM Predictions to File
# ------------------------

output_file = 'pred_deepfeat.txt'
with open(output_file, 'w') as f:
    for prediction in y_pred_svm:
        f.write(f"{prediction}\n")

# --------------------------------
#   Text Sequence Dataset
# --------------------------------

class MLModel():
    def init(self) -> None:
        pass
    
class TextSeqModel(MLModel):
    def __init__(self) -> None:
        super().init()
    
    def predict(self, X):
        # Load the saved LSTM model
        self.model = load_model('lstm_trained_on_100_percent_lr=0.01_bs=128.keras')
        
        # Preprocess the input data
        X_ = np.zeros((len(X), len(X[0])))
        for i in range(len(X)):
            for j in range(len(X[0])):
                X_[i][j] = X[i][j]
                
        # Standardize the features using saved scaler
        scaler = load('scaler.joblib')
        X_test = scaler._transform(X_)
        
        # Predict using the model
        p = self.model.predict(X_test)
        return (p > 0.5).astype(int)
    
def save_predictions_to_file(predictions, filename):
    # Save predictions to a text file
    with open(filename, 'w') as f:
        for pred in predictions:
            f.write(f"{int(pred)}\n")

# Load the test dataset (text sequence data)
test_seq_X = pd.read_csv(test_path_text)['input_str'].tolist()

# Initialize and make predictions with the TextSeqModel
text_model = TextSeqModel()
pred_text = text_model.predict(test_seq_X)

# Save predictions to file
save_predictions_to_file(pred_text, "pred_text.txt")

# --------------------------------
#   Combined Dataset (Deep Features)
# --------------------------------

# Load and preprocess the test data
DATA = np.load(test_path_deep, allow_pickle=True)
x_test = DATA['features']
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

# Load the saved SVM model and make predictions
with open('svm_max.pkl', 'rb') as f:
    svm_model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
y_pred_svm = svm_model.predict(scaler.transform(x_test))

# Save combined predictions to a text file
output_file = 'pred_combined.txt'
with open(output_file, 'w') as f:
    for prediction in y_pred_svm:
        f.write(f"{prediction}\n")
