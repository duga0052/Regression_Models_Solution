import pandas as pd
import logging

def load_and_preprocess_data(data_path):
    try:
        df = pd.read_csv(data_path)
        logging.info("Data loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error in processing data: {e}")
        raise