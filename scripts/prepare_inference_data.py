import os
import pandas as pd
from src.data.loader import DataLoader
from src.data.preprocessor import BWFPreprocessor
from src.data.feature_engine import FeatureEngineer
from src.config import settings

def main():
    print("Loading and preprocessing data...")
    loader = DataLoader()
    raw_df = loader.load("bwf_cleaned.csv")
    
    preprocessor = BWFPreprocessor()
    clean_df = preprocessor.fit_transform(raw_df)
    
    fe = FeatureEngineer()
    os.makedirs("data/processed", exist_ok=True)
    
    # Process all combinations
    for draw in settings.DRAWS:
        for region in ["Global", "Asia", "Europe"]:
            print(f"Engineering features for {draw} / {region}...")
            features_df = fe.transform(clean_df, draw=draw, region=region)
            
            if not features_df.empty:
                latest_date = features_df["date"].max()
                latest_features = features_df[features_df["date"] == latest_date].copy()
                
                out_path = f"data/processed/latest_features_{draw}_{region}.csv"
                latest_features.to_csv(out_path, index=False)
                print(f"Saved {out_path}, shape: {latest_features.shape}")
            else:
                print(f"Skipping {draw}/{region} (no data)")

if __name__ == "__main__":
    main()
