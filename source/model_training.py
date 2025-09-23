# Libraries for data manipulation
import pandas as pd
import numpy as np

# Libraries for NN training 
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler, SMOTE




# ----------DATA MANIPULATION-------------
# Pull clean dataset 
df = pd.read_csv('C:/Users/cqalv/Documents/Projects/Fraud_Detection/data/creditcard_clean.csv')

# Drop nonessential columns from dataset
df.drop(['FraudLabel', 'HourOfDay', 'TransactionDay'], axis=1, inplace=True)

# Train Test Split
X = df.drop('IsFraud', axis=1).values
y = df['IsFraud'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=29)

# Count of fraud vs nonfraud
nonfraud_count = (y_train == 0).sum()
fraud_count = (y_train == 1).sum()

# Fraud weight and ratio of class imbalance
fraud_weight = nonfraud_count / fraud_count

# Oversample the dataset to make up for lack of fraud

# SMOTE
sm = SMOTE(random_state = 42)
X_sample, y_sample = sm.fit_resample(X_train,y_train)

# Convert features to tensors
X_sample = torch.FloatTensor(X_train)
y_sample = torch.FloatTensor(y_train).view(-1, 1)

X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).view(-1, 1)

train_dataset1 = TensorDataset(X_sample, y_sample)
train_loader1 = DataLoader(train_dataset1, batch_size=512, shuffle=True)

# Random Oversampling
oversample = RandomOverSampler(sampling_strategy='minority')
X_over, y_over = oversample.fit_resample(X_train, y_train)

# Convert features to tensors
X_over = torch.FloatTensor(X_train)
y_over = torch.FloatTensor(y_train).view(-1, 1)

train_dataset2 = TensorDataset(X_over, y_over)
train_loader2 = DataLoader(train_dataset2, batch_size=512, shuffle=True)


# -------------TRAINING---------------------
from neural_net import FraudDetectionModel

# Instance of model
model = FraudDetectionModel(input_attributes=X_train.shape[1])

# Measure Prediction Error with BCE
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([fraud_weight]))

# Optimizer Adam Algorithm, lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop for Model
# An Epoch is one pass of all training data through model

epochs = 50
losses = []

for epoch in range(epochs):
    for X_batch, y_batch in train_loader2:

        # Prediction and loss calculation
        y_prediction = model(X_batch)
        loss = criterion(y_prediction, y_batch)

        # Back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print every 10 loops
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, loss: {loss.item():.4f}")
        losses.append(loss.item())






