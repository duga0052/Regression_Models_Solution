import os
import logging
from src.data.load_dataset import load_and_preprocess_data
from src.feature.build_features import create_features
from src.model.train_model import split_data, train_linear_reg, train_decision_tree, train_random_forest, save_model, load_model
from src.model.predict_model import make_predictions, evaluate_model
from src.visualization.visualize import plot_decision_tree

# Ensure the log file exists
log_file_exists = os.path.exists('app.log')
if not log_file_exists:
    with open('app.log', 'w') as f:
        f.write('Log file created.\n')

# Configure logging
def setup_logging():
    logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')

setup_logging()


if __name__ == "__main__":
    try:
        # Load and preprocess the data
        data_path = "src/data/final.csv"
        df = load_and_preprocess_data(data_path)
        logging.info("Data loaded and preprocessed successfully.")

        # Prepare data
        x, y = create_features(df)
        logging.info("Features created successfully.")

        # Split data
        x_train, x_test, y_train, y_test = split_data(df, x, y)
        logging.info("Data split into training and testing sets successfully.")

        # Train and evaluate Linear Regression model
        model = train_linear_reg(x_train, y_train)
        train_mae, test_mae = evaluate_model(model, x_train, x_test, y_train, y_test)
        logging.info(f'Linear Regression - Train MAE: {train_mae}, Test MAE: {test_mae}')
        print('Linear Regression - Train MAE: ', train_mae)
        print('Linear Regression - Test MAE: ', test_mae)

        # Train and evaluate Decision Tree model
        dt_model = train_decision_tree(x_train, y_train)
        dt_train_mae, dt_test_mae = evaluate_model(dt_model, x_train, x_test, y_train, y_test)
        logging.info(f'Decision Tree - Train MAE: {dt_train_mae}, Test MAE: {dt_test_mae}')
        print('Decision Tree - Train MAE: ', dt_train_mae)
        print('Decision Tree - Test MAE: ', dt_test_mae)

        # Plot Decision Tree
        plot_decision_tree(dt_model, x.columns)
        logging.info("Decision Tree plotted successfully.")
        
        # Train and evaluate Random Forest model
        rf_model = train_random_forest(x_train, y_train)
        rf_train_mae, rf_test_mae = evaluate_model(rf_model, x_train, x_test, y_train, y_test)
        logging.info(f'Random Forest - Train MAE: {rf_train_mae}, Test MAE: {rf_test_mae}')
        print('Random Forest - Train MAE: ', rf_train_mae)
        print('Random Forest - Test MAE: ', rf_test_mae)

        # Save the Random Forest model
        save_model(rf_model, 'RE_Model.pkl')
        logging.info("Random Forest model saved successfully.")

        # Load the saved model and make a prediction
        loaded_model = load_model('RE_Model.pkl')
        sample_prediction = loaded_model.predict([[2012, 216, 74, 1, 1, 618, 2000, 600, 1, 0, 0, 6, 0]])
        logging.info(f'Sample prediction: {sample_prediction}')
        print('Sample prediction:', sample_prediction)
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")