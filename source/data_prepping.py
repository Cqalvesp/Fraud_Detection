# Data preparation for ML Model
import pandas as pd
import numpy as np


def data_prep():
    # Load dataset
    df_Clean = pd.read_csv('C:/Users/cqalv/Documents/Projects/Fraud_Detection/data/creditcard.csv')

    # Rename columns for easy interpretation
    df_Clean.rename(columns = {'Time' : 'TransactionTime', 'Amount' : 'TransactionAmount', 'Class' : 'IsFraud'}, inplace=True)

    # Add column for Hour of day
    df_Clean['HourOfDay'] = (df_Clean['TransactionTime'] / 3600) % 24
    # Add column for Fraud Label
    df_Clean['FraudLabel'] = np.where(df_Clean['IsFraud'] == 1, 'Fraud', 'Not Fraud')
    # Add column for day of transaction
    df_Clean['TransactionDay'] = (df_Clean['TransactionTime'] // 86400) + 1
    
    return df_Clean

def aggregate_data(df):
    pass

    

if __name__ == "__main__":
    print(data_prep())
    print(aggregate_data(data_prep()))
