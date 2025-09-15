import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time as time
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

def T_stage_by_size(size):
    if size == 0:
        return 0
    if size > 0 and size <= 20:
        return 1
    if size > 20 and size <= 50:
        return 2
    if size > 50:
        return 3
        
def mapping(row, dictionary):
    return dictionary[row]

# FEED FORWARD NEURAL NETWORK 
class DrugEncoder(nn.Module):
    def __init__(self, input_dim=5):  # 5 molecular descriptors
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

    def forward(self, x):
        return self.encoder(x.float())

# SIMGPLE FEED FORWARD NEURAL NETWORK 
class PatientEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

    def forward(self, x):
        return self.encoder(x.float())

# REGRESSION PREDICTION WITH SIMPLE FEED FORWARD NEURAL NETWORK

class DrugResponsePredictor(nn.Module):
    def __init__(self, drug_input_dim, patient_input_dim):
        super().__init__()
        self.drug_encoder = DrugEncoder(drug_input_dim)
        self.patient_encoder = PatientEncoder(patient_input_dim)
        self.fc = nn.Sequential(
            nn.Linear(32 + 32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # IC50 prediction
        )

    def forward(self, drug_feat, patient_feat):
        drug_vec = self.drug_encoder(drug_feat)
        patient_vec = self.patient_encoder(patient_feat)
        combined = torch.cat([drug_vec, patient_vec], dim=1)
        return self.fc(combined).squeeze(-1)
###

# Evaluating model
def evaluate_model(model, drug_features, patient_features, true_ic50s):
    # Convert to numpy 
    if hasattr(drug_features, 'to_numpy'):
        drug_features = drug_features.to_numpy()
    if hasattr(patient_features, 'to_numpy'):
        patient_features = patient_features.to_numpy()
    if hasattr(true_ic50s, 'to_numpy'):
        true_ic50s = true_ic50s.to_numpy()

    # Convert to tensors
    drug_tensor = torch.tensor(drug_features).float()
    patient_tensor = torch.tensor(patient_features).float()
    true_ic50_tensor = torch.tensor(true_ic50s).float()

    # Inference
    model.eval()
    with torch.no_grad():
        preds = model(drug_tensor, patient_tensor).cpu().numpy()
        targets = true_ic50_tensor.cpu().numpy()

    # Metrics
    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)

    return mse, mae, r2

# Visualizing training loss
def epochs_vs_loss_with_cells(epochs, loss):
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, loss, alpha=0.6, color = "hotpink")

    plt.xlabel("Epochs", fontsize=15)
    plt.ylabel("Loss (MSE)", fontsize=15)
    plt.title("Epochs vs. Loss during training \nWith Cell Line information")
    plt.tight_layout()
    plt.savefig("500epochs.png")
    plt.show()


def epochs_vs_loss_without_cells(epochs, loss):
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, loss, alpha=0.6, color = "hotpink")

    plt.xlabel("Epochs", fontsize=15)
    plt.ylabel("Loss (MSE)", fontsize=15)
    plt.title("Epochs vs. Loss during training \nExcluding Cell Line information")
    plt.tight_layout()
    plt.show()

def epochs_vs_loss_baseline(epochs, loss):
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, loss, alpha=0.6, color = "hotpink")

    plt.xlabel("Epochs", fontsize=15)
    plt.ylabel("Loss (MSE)", fontsize=15)
    plt.title("Epochs vs. Loss during training \nExcluding Cell Line information - BASELINE MODEL")
    plt.tight_layout()
    plt.show()


# Visualizing results
def plot_pred_vs_true_with_cells(true, preds):
    plt.figure(figsize=(8, 6))
    plt.scatter(true, preds, alpha=0.6, color = "hotpink")
    
    plt.plot([true.min(), true.max()], [true.min(), true.max()], 'r--', label='Ideal', color = "Black")

    slope, intercept = np.polyfit(true, preds, 1)
    plt.plot(true, slope * true + intercept, 'g-', label='Best Fit Line')

    plt.xlabel("True IC50", fontsize=15)
    plt.ylabel("Predicted IC50", fontsize=15)
    plt.title("True vs. Predicted IC50 values\nModel trained with 300 epochs\nWith Cell Line information")
    plt.legend()
    plt.tight_layout()
    plt.savefig("WithCellLineInformation.png")
    plt.show()

# Visualizing results
def plot_pred_vs_true_without_cells(true, preds):
    plt.figure(figsize=(8, 6))
    plt.scatter(true, preds, alpha=0.6, color = "hotpink")
    
    plt.plot([true.min(), true.max()], [true.min(), true.max()], 'r--', label='Ideal', color = "Black")

    slope, intercept = np.polyfit(true, preds, 1)
    plt.plot(true, slope * true + intercept, 'g-', label='Best Fit Line')

    plt.xlabel("True IC50", fontsize=15)
    plt.ylabel("Predicted IC50", fontsize=15)
    plt.title("True vs. Predicted IC50 values\nModel trained with 300 epochs\nExcluding Cell Line information")
    plt.legend()
    plt.tight_layout()
    plt.savefig("WithoutCellLineInformation.png")
    plt.show()

# Visualizing results
def plot_pred_vs_true_baseline(true, preds):
    plt.figure(figsize=(8, 6))
    plt.scatter(true, preds, alpha=0.6, color = "hotpink")
    
    plt.plot([true.min(), true.max()], [true.min(), true.max()], 'r--', label='Ideal', color = "Black")

    slope, intercept = np.polyfit(true, preds, 1)
    plt.plot(true, slope * true + intercept, 'g-', label='Best Fit Line')

    plt.xlabel("True IC50", fontsize=15)
    plt.ylabel("Predicted IC50", fontsize=15)
    plt.title("True vs. Predicted IC50 values\nModel trained with 300 epochs\nExcluding Cell Line information - BASELINE MODEL")
    plt.legend()
    plt.tight_layout()
    plt.savefig("BaselineResults.png")
    plt.show()

# Cross validation 
from sklearn.model_selection import KFold
def compute_CV_error(X_train, Y_train):
    '''
    Split the training data into 4 subsets.
    For each subset, 
        - Fit a model holding out that subset.
        - Compute the MSE on that subset (the validation set).
    You should be fitting 4 models in total.
    Return the average MSE of these 4 folds.

    Args:
        model: An sklearn model with fit and predict functions. 
        X_train (DataFrame): Training data.
        Y_train (DataFrame): Label.
    
    Return:
        The average validation MSE for the 4 splits.
    '''
    kf = KFold(n_splits=10)
    validation_errors = []
    validation_r2 = []
    training_errors = []
    
    for train_idx, valid_idx in kf.split(X_train):
        # split the data
        split_X_train, split_X_valid = X_train.iloc[train_idx], X_train.iloc[valid_idx] 
        split_Y_train, split_Y_valid = Y_train.iloc[train_idx], Y_train.iloc[valid_idx]

        ##  Extract drug features  ##
        # Select drug descriptor columns
        drug_features_train = split_X_train[['Molecular Mass', 'LogP', 'NumHDonors', 'NumHAcceptors', 'TPSA']]
        patient_features_train = split_X_train.drop(columns=drug_features_train.columns)

        # Convert to tensors
        # drug_tensor = torch.tensor(drug_features_train).float()
        drug_tensor = torch.tensor(drug_features_train.to_numpy()).float()
        patient_tensor = torch.tensor(patient_features_train.to_numpy()).float()
        ic50_tensor = torch.tensor(split_Y_train.to_numpy()).float()
        
        # Assign optimizer 
        model = DrugResponsePredictor(drug_input_dim=5, patient_input_dim=patient_tensor.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        
        # Fit the model on the training split
        for epoch in range(300):  
            optimizer.zero_grad()
            preds = model(drug_tensor, patient_tensor)
            loss = loss_fn(preds, ic50_tensor)
            loss.backward()
            optimizer.step()
        training_errors.append(loss.item())
        # Compute the MSE on the validation split
        drug_features_val = split_X_valid[['Molecular Mass', 'LogP', 'NumHDonors', 'NumHAcceptors', 'TPSA']]
        patient_features_val = split_X_valid.drop(columns=drug_features_val.columns)

        mse, mae, r2 = evaluate_model(model, drug_features_val, patient_features_val, split_Y_valid)

        validation_errors.append(mse)
        validation_r2.append(r2)
        
    return training_errors, validation_errors, validation_r2

def Kfold_CV_plot_with_cells(k_fold_cv):
    plt.figure(figsize=(8, 6))
    plt.plot([i for i in range(10)], k_fold_cv[1], alpha=0.6, label='validation MSE', color = "hotpink")
    plt.plot([i for i in range(10)], k_fold_cv[0], alpha=0.6, label='training MSE', color = "purple")
    plt.axhline(y=np.mean(k_fold_cv[1]), color='hotpink', linestyle='--', label='Average validation loss')
    plt.axhline(y=np.mean(k_fold_cv[0]), color='purple', linestyle='--', label='Average training loss')

    plt.xlabel("Fold", fontsize=15)
    plt.xticks(ticks=range(10), labels=range(1, 11))
    plt.ylabel("Evaluation", fontsize=15)
    plt.title("K Fold cross-validation (k=10)\nProof of well generalizied model \n(i.e. MODEL NOT OVER-FITTED)\nWith Cell Line information")
    plt.legend()
    plt.tight_layout()
    plt.show()

def Kfold_CV_plot_without_cells(k_fold_cv):
    plt.figure(figsize=(8, 6))
    plt.plot([i for i in range(10)], k_fold_cv[1], alpha=0.6, label='validation MSE', color = "hotpink")
    plt.plot([i for i in range(10)], k_fold_cv[0], alpha=0.6, label='training MSE', color = "purple")
    plt.axhline(y=np.mean(k_fold_cv[1]), color='hotpink', linestyle='--', label='Average validation loss')
    plt.axhline(y=np.mean(k_fold_cv[0]), color='purple', linestyle='--', label='Average training loss')

    plt.xlabel("Fold", fontsize=15)
    plt.xticks(ticks=range(10), labels=range(1, 11))
    plt.ylabel("Evaluation", fontsize=15)
    plt.title("K Fold cross-validation (k=10)\nProof of well generalizied model \n(i.e. MODEL NOT OVER-FITTED)\nExcluding Cell Line information")
    plt.legend()
    plt.tight_layout()
    plt.show()

def Kfold_CV_plot_baseline(k_fold_cv):
    plt.figure(figsize=(8, 6))
    plt.plot([i for i in range(10)], k_fold_cv[1], alpha=0.6, label='validation MSE', color = "hotpink")
    plt.plot([i for i in range(10)], k_fold_cv[0], alpha=0.6, label='training MSE', color = "purple")
    plt.axhline(y=np.mean(k_fold_cv[1]), color='hotpink', linestyle='--', label='Average validation loss')
    plt.axhline(y=np.mean(k_fold_cv[0]), color='purple', linestyle='--', label='Average training loss')

    plt.xlabel("Fold", fontsize=15)
    plt.xticks(ticks=range(10), labels=range(1, 11))
    plt.ylabel("Evaluation", fontsize=15)
    plt.title("K Fold cross-validation (k=10)\nProof of well generalizied model \n(i.e. MODEL NOT OVER-FITTED)\nExcluding Cell Line information - BASELINE MODEL")
    plt.legend()
    plt.tight_layout()
    plt.show()