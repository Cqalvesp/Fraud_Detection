# Script to visualize important traits of the dataset

import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('C:/Users/cqalv/Documents/Projects/Fraud_Detection/data/creditcard_clean.csv')

def class_imbalance():
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

    plt.show()
    return





