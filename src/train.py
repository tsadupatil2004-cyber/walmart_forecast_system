import yaml
import joblib
import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor

from preprocessing import DataPreprocessor
from features import FeatureEngineer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 1. Preprocess
    preprocessor = DataPreprocessor(
        config['data']['train_path'],
        config['data']['features_path'],
        config['data']['stores_path']
    )
    df = preprocessor.load_and_merge()

    # 2. Features
    df, feature_cols = FeatureEngineer.create_features(df)

    X = df[feature_cols]
    y = df['Weekly_Sales']

    # 3. Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['model']['test_size'], random_state=config['model']['random_state']
    )

    # 4. Train XGBoost Model
    logging.info("Training XGBoost Regressor model...")
    model = XGBRegressor(
        n_estimators=config['model']['n_estimators'],
        learning_rate=0.1,
        max_depth=6,
        random_state=config['model']['random_state'],
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 5. Evaluate
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    logging.info(f"Model Performance -> MAE: {mae:.2f} | RMSE: {rmse:.2f}")

    # 6. Save Artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, config['model']['output_path'])
    logging.info(f"Trained model saved to {config['model']['output_path']}")

    # Save sample for dashboard UI
    os.makedirs("data", exist_ok=True)
    df.sample(n=min(30000, len(df)), random_state=42).to_csv(config['data']['sample_output'], index=False)
    logging.info("Processed data sample saved.")

if __name__ == "__main__":
    main()