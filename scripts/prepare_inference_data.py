import os
import pandas as pd
from src.data.loader import DataLoader
from src.data.preprocessor import BWFPreprocessor
from src.data.feature_engine import FeatureEngineer

def main():
    print("Loading raw data...")
    loader = DataLoader()
    raw_df = loader.load("bwf_cleaned.csv")
    
    print("Preprocessing data...")
    preprocessor = BWFPreprocessor()
    clean_df = preprocessor.fit_transform(raw_df)
    
    print("Engineering features for MS / Asia...")
    fe = FeatureEngineer()
    features_df = fe.transform(clean_df, draw="MS", region="Asia")
    
    print("Extracting latest rows...")
    latest_date = features_df["date"].max()
    latest_features = features_df[features_df["date"] == latest_date].copy()
    
    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/latest_features_MS_Asia.csv"
    latest_features.to_csv(out_path, index=False)
    print(f"Saved {out_path}, shape: {latest_features.shape}")

if __name__ == "__main__":
    main()
