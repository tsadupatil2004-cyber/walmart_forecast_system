import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataPreprocessor:
    def __init__(self, train_path, features_path, stores_path):
        self.train_path = train_path
        self.features_path = features_path
        self.stores_path = stores_path

    def load_and_merge(self):
        logging.info("Loading CSV datasets...")
        train = pd.read_csv(self.train_path)
        features = pd.read_csv(self.features_path)
        stores = pd.read_csv(self.stores_path)

        logging.info("Merging datasets on Store, Date, and IsHoliday...")
        df = train.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left')
        df = df.merge(stores, on=['Store'], how='left')

        # Clean missing values
        df['Date'] = pd.to_datetime(df['Date'])
        for col in ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']:
            df[col] = df[col].fillna(0.0)
        
        df['CPI'] = df['CPI'].fillna(df['CPI'].median())
        df['Unemployment'] = df['Unemployment'].fillna(df['Unemployment'].median())

        # Encode categorical store types
        type_mapping = {'A': 1, 'B': 2, 'C': 3}
        df['Type_Encoded'] = df['Type'].map(type_mapping)
        df['IsHoliday'] = df['IsHoliday'].astype(int)

        logging.info("Data preprocessing completed successfully.")
        return df