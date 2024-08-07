Regression_Models_Solution

==============================

Restructuring your ML code for production deployment

Project Organization

------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── model            <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── reference         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── report            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── load_dataset.py
    │   │
    │   ├── feature       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── model         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

Purpose
This project aims to predict real estate prices based on various features using machine learning techniques. The dataset includes information about properties such as size, location, number of bedrooms, and other relevant features. The project involves data loading, preprocessing, feature engineering, model training, evaluation, and visualization.

------------

How to Run This Code
 - Ensure you have Python installed on your system along with the required packages.
 - Place final.csv in the src/data/ directory.
 - Run the main.py script in a Python environment.

-------------

Dependencies

The following libraries are required:
pandas: For data manipulation and analysis
numpy: For numerical operations
matplotlib: For plotting graphs
seaborn: For data visualization
scikit-learn: For machine learning algorithms and evaluation metrics
pickle: For saving and loading machine learning models
logging: For logging errors and information
Ensure they are installed using pip:

---------------

Ensure they are installed using pip:

pip install pandas numpy matplotlib seaborn scikit-learn

-------------

Detailed Steps
1. Data Loading
The dataset is loaded from a CSV file named final.csv using the load_and_preprocess_data function from load_dataset.py. This function reads the data into a pandas DataFrame and performs initial preprocessing.
2. Data Exploration
Initial Inspection: Display the first few rows of the dataset to understand its structure and contents.
Check for Missing Values: Identify any missing values in the dataset to plan for imputation.
3. Data Cleaning and Preprocessing
Impute Missing Values: Handle missing values by imputing them with appropriate statistics (mode or median).
Drop Unnecessary Columns: Remove columns that are not needed for the analysis.
Create Features: Convert categorical variables into dummy variables to prepare the data for machine learning algorithms.
4. Feature Engineering
Separate Features and Target Variable: Split the dataset into input features (x) and the target variable (y), which is price.
Data Splitting: Divide the dataset into training and testing sets using the split_data function.
5. Model Training
Linear Regression: Train a linear regression model using the train_linear_reg function.
Decision Tree: Train a decision tree model using the train_decision_tree function.
Random Forest: Train a random forest model using the train_random_forest function.
6. Model Evaluation
Evaluate Models: Measure the performance of the models using Mean Absolute Error (MAE) on both training and testing sets using the evaluate_model function.
Cross-Validation: Perform k-fold cross-validation to ensure the model's robustness and generalizability.
7. Model Interpretation
Plot Decision Tree: Visualize the decision tree model using the plot_decision_tree function.
8. Model Saving and Loading
Save Model: Save the trained model to a file using the save_model function.
Load Model: Load the saved model from a file using the load_model function.

--------------------
Conclusion
This project demonstrates a complete workflow for predicting real estate prices using machine learning. It covers data preprocessing, feature engineering, model training, evaluation, and visualization. The models are evaluated using Mean Absolute Error (MAE) to ensure robustness and reliability. The linear regression, decision tree, and random forest models provide a comprehensive approach to understanding and predicting real estate prices based on property features.

---------------------

Steps to Push code from VS code to Github.
First authenticate your githib account and integrate with VS code. Click on the source control icon and complete the setup.
1. Click terminal and open new terminal
2. git config --global user.name "Swapnilin"
3. git config --global user.email swapnilforcat@gmail.com
4. git init
5. git add .
6. git commit -m "Your commit message"