import logging
from sklearn.metrics import mean_absolute_error

def make_predictions(model, x):
    try:
        return model.predict(x)
    except Exception as e:
        logging.error(f"Error in making predictions: {e}")
        raise
        
def evaluate_model(model, x_train, x_test, y_train, y_test):
    try:
        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        logging.info("Model evaluated successfully.")
        return train_mae, test_mae
    except Exception as e:
        logging.error(f"Error in evaluating model: {e}")
        raise