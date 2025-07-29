# Script to visualize important traits of the dataset

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import matplotlib as mlp
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

df = pd.read_csv('C:/Users/cqalv/Documents/Projects/Fraud_Detection/data/creditcard.csv')



