from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import logging, pickle

def split_data(df, X, y):
    try:
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234, stratify=df['beds'])
        logging.info("Data split successfully.")
        return x_train, x_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error in splitting data: {e}")
        raise

def train_linear_reg(x_train, y_train):
    try:
        model = LinearRegression()
        model.fit(x_train, y_train)
        logging.info("Linear Regression model trained successfully.")
        return model
    except Exception as e:
        logging.error(f"Error in training Linear Regression model: {e}")
        raise

def train_decision_tree(x_train, y_train, max_depth=3, max_features=10, random_state=567):
    try:
        model = DecisionTreeRegressor(max_depth=max_depth, max_features=max_features, random_state=random_state)
        model.fit(x_train, y_train)
        logging.info("Decision Tree model trained successfully.")
        return model
    except Exception as e:
        logging.error(f"Error in training Decision Tree model: {e}")
        raise
        
def train_random_forest(x_train, y_train, n_estimators=200, criterion='absolute_error'):
    try:
        model = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion)
        model.fit(x_train, y_train)
        logging.info("Random Forest model trained successfully.")
        return model
    except Exception as e:
        logging.error(f"Error in training Random Forest model: {e}")
        raise
        
def save_model(model, file_name):
    try:
        with open(file_name, 'wb') as file:
            pickle.dump(model, file)
        logging.info("Model saved successfully.")
    except Exception as e:
        logging.error(f"Error in saving model: {e}")
        raise
        
def load_model(file_name):
    try:
        with open(file_name, 'rb') as file:
            model = pickle.load(file)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error in loading model: {e}")
        raise