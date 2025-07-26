# Loading fraud data from AWS S3
import boto3, os

# Cleaning and preprocessing fraud data
import pandas as pd

# AWS S3 Info
Bucket = 'cqalvesp-data-ingestion'
Key = 'fraud_data/raw/creditcard.csv'
Dest = "data/creditcard.csv"

s3 = boto3.client('s3')
s3.download_file(Bucket, Key, Dest)

def download_data():
    s3 = boto3.client('s3')
    s3.download_file(Bucket, Key, Dest)
    print(f"Downloaded credit card data to {Dest}")

    return

if __name__ == "__main__":
    download_data()