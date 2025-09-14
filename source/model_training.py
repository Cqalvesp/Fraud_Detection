# Libraries for data manipulation
import pandas as pd
import numpy as np

# Libraries for NN training 
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from neural_net import FraudDetectionModel
import matplotlib.pyplot as plt





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

# Convert features to tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).view(-1, 1)

X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).view(-1, 1)

# Count of fraud vs nonfraud
nonfraud_count = (y_train == 0).sum()
fraud_count = (y_train == 1).sum()

# Fraud weight and ratio of class imbalance
fraud_weight = nonfraud_count / fraud_count



# -------------TRAINING---------------------
# Instance of model
model = FraudDetectionModel(input_attributes=X_train.shape[1])

# Measure Prediction Error with BCE
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([fraud_weight]))

# Optimizer Adam Algorithm, lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop for Model
# An Epoch is one pass of all training data through model

epochs = 100
losses = []

for i in range(epochs):
    # Predicted results from data
    y_prediction = model(X_train) 

    # Measure prediction error
    # Predicted values vs training values
    loss = criterion(y_prediction, y_train)

    # Track losses
    losses.append(loss.detach().numpy())

    # Print every 10 loops
    if i % 10 == 0:
        print(f'Epoch: {i} and loss: {loss}')

    # Back propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step

for name, param in model.named_parameters():
    if param.grad is not None:
        print(name, param.grad.norm().item())

# --------------GRAPHING----------------
# Graph Neural Network Training Loss over time
plt.plot(range(epochs), losses)
plt.ylabel("loss/error")
plt.xlabel("epoch")
plt.show()






