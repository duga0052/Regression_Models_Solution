import pandas as pd
import logging

def create_features(df):
    try:
        x = df.drop('price', axis=1)
        y = df['price']
        logging.info("Features created successfully.")
        return x, y
    except Exception as e:
        logging.error(f"Error in creating features: {e}")
        raise