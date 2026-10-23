import pandas as pd
import logging

class FeatureEngineer:
    @staticmethod
    def create_features(df):
        logging.info("Extracting time-series and calendar features...")
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
        df['Day'] = df['Date'].dt.day
        
        feature_cols = [
            'Store', 'Dept', 'IsHoliday', 'Size', 'Type_Encoded', 
            'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 
            'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 
            'Unemployment', 'Year', 'Month', 'Week', 'Day'
        ]
        
        return df, feature_cols