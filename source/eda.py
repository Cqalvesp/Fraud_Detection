# Script to visualize important traits of the dataset

import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('C:/Users/cqalv/Documents/Projects/Fraud_Detection/data/creditcard_clean.csv')

def class_imbalance(df):
    # Calculate percentages
    fraud_counts = df['IsFraud'].value_counts(normalize=True) * 100
    fraud_df = fraud_counts.rename_axis('IsFraud').reset_index(name='Percentage')

    # Seaborn style
    sns.set_style("whitegrid")

    # Plot
    sns.barplot(data=fraud_df, x='IsFraud', y='Percentage', palette=["darkblue", "darkred"])

    # Customize labels
    plt.xticks([0, 1], ["Not Fraud", "Fraud"])
    plt.ylabel("Percentage of Transactions (%)")
    plt.xlabel("")
    plt.title("Fraud vs. Not Fraud (Percentage)")
    plt.ylim(0, 100)

    # Show values on bars
    for index, row in fraud_df.iterrows():
        plt.text(row.name, row.Percentage + 1, f"{row.Percentage:.2f}%", ha='center')

    plt.savefig("C:/Users/cqalv/Documents/Projects/Fraud_Detection/visualizations/class_imbalance.pdf", format="pdf")
    return

def heatmap(df):
    corr = df.corr(numeric_only=True)

    sns.set_style("white")
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, cmap="coolwarm", annot=False, cbar=True)
    plt.title("Feature Correlation Heatmap", fontsize=16)
    plt.tight_layout()
    
    plt.savefig("C:/Users/cqalv/Documents/Projects/Fraud_Detection/visualizations/heatmap.pdf", format="pdf")
    print(
        corr["IsFraud"]
        .sort_values(ascending=False)
        .to_string()
    )
    return

def fraud_frequency(df):
    hourly = (
    df.groupby("HourOfDay")
      .agg(total=("IsFraud", "count"), fraud=("IsFraud", "sum"))
      .reset_index()
    )
    hourly["fraud_rate_pct"] = hourly["fraud"] / hourly["total"] * 100

    plt.figure(figsize=(10, 4))
    sns.barplot(data=hourly, x="HourOfDay", y="fraud_rate_pct", color="salmon")
    plt.title("Fraud Rate (%) by Hour of Day")
    plt.xlabel("Hour of Day (0–23)")
    plt.ylabel("Fraud Rate (%)")
    plt.tight_layout()
    plt.savefig("C:/Users/cqalv/Documents/Projects/Fraud_Detection/visualizations/fraud_frequency.pdf", format="pdf")
    return

if __name__ == "__main__":
    data = pd.read_csv('C:/Users/cqalv/Documents/Projects/Fraud_Detection/data/creditcard_clean.csv')

    class_imbalance(data)
    heatmap(data)
    fraud_frequency(data)





