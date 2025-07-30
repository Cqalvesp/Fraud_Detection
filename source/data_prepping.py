# Data preparation for ML Model
import pandas as pd
import numpy as np


def data_prep():
    # Load dataset
    df_Clean = pd.read_csv('C:/Users/cqalv/Documents/Projects/Fraud_Detection/data/creditcard.csv')

    # Rename columns for easy interpretation
    df_Clean.rename(columns = {'Time' : 'TransactionTime', 'Amount' : 'TransactionAmount', 'Class' : 'IsFraud'}, inplace=True)

    # Add column for Hour of day
    df_Clean['HourOfDay'] = pd.NA
    # Add column for Fraud Label
    df_Clean['FraudLabel'] = pd.NA
    # Add column for day of transaction
    df_Clean['TransactionDay'] = pd.NA

    for index, row in df_Clean.iterrows():
        if row['IsFraud'] == 0:
            row['FraudLabel'] = 'Not Fraud'
        elif row['IsFraud'] == 1:
            row['FraudLabel'] = 'Fraud'

        row['TransactionDay'] = row['TransactionTime'] // 86400
        row['HourOfDay'] = (row['TransactionTime'] / 3600) % 24
    
    return df_Clean

    

if __name__ == "__main__":
    print(data_prep().head())
