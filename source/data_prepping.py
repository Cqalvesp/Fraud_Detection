# Data preparation for ML Model
import pandas as pd
import numpy as np


def clean_raw_data():
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

def aggregate_data(df, view_type):

    def fraud_summary(grouped_df, view_type):

        summary = grouped_df.agg(
            TotalTransactions=("IsFraud", "count"),
            FraudTransactions=("IsFraud", "sum")
        ).reset_index()
        summary["FraudRate (%)"] = (
            summary["FraudTransactions"] / summary["TotalTransactions"] * 100
        )
        summary["ViewType"] = view_type

        return summary


    fraud_by_day = fraud_summary(df.groupby("TransactionDay"), "By Day")


    fraud_by_hour = fraud_summary(df.groupby("HourOfDay"), "By Hour")


    bins = [0, 10, 100, 500, float("inf")]
    labels = ["0-10", "10-100", "100-500", "500+"]
    df["AmountBin"] = pd.cut(df["TransactionAmount"], bins=bins, labels=labels, right=False)

    fraud_by_amount = fraud_summary(df.groupby("AmountBin"), "By Amount Bin")


    combined_summary = pd.concat([fraud_by_day, fraud_by_hour, fraud_by_amount], ignore_index=True)


    combined_summary.to_csv("data/creditcard_summary.csv", index=False)

    

if __name__ == "__main__":
    print(clean_raw_data())
    print(aggregate_data(clean_raw_data()))
