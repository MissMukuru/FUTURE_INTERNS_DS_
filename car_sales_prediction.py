import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib


def encode_categorical_features(df):
    """Encoding the categorical features of the data"""
    df = pd.get_dummies(df, drop_first=True)  # Use one-hot encoding for categorical features
    return df


def define_features_target_variables(df):
    """Defining the feature and target variables"""
    X = df.drop('car purchase amount', axis=1)
    y = df['car purchase amount']
    
    return X, y


def split_data(X, y):
    """Splitting the data"""
    return train_test_split(X, y, random_state=42, test_size=0.3)


def scale_data(X_train, X_test):
    """Fitting the scaler onto the training data and applying the transformations to the test set"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Apply transformation to test set only
    
    return X_train_scaled, X_test_scaled, scaler


def initialize_model():
    """Creating the model instances"""
    models = {
        'LinearRegression': LinearRegression(),
        'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42),
        'RandomForestRegressor': RandomForestRegressor(random_state=42)
    }
    
    return models


def train_and_predict(models, X_train, X_test, y_train):
    """Loop through all the models, train them via the training set, and predict via the testing set"""
    prediction = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        prediction[name] = y_pred
        
    return prediction


def evaluate_model_performance(models, prediction, y_test):
    """Get the model metrics, print the performance metrics and save the model with the best r2_score"""
    
    metrics = []
    best_r2_score = -np.inf
    best_model = None
    
    for name, y_pred in prediction.items():
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f'{name} - Mean Squared Error: {mse}, R2 Score: {r2}')
        
        metrics.append({
            'Name': name,
            'Mean Squared Error': mse,
            'R2 Score': r2
        })
        
        if r2 > best_r2_score:
            best_r2_score = r2
            best_model = models[name]
            
    metrics_df = pd.DataFrame(metrics)
    
    return best_model, metrics_df


def main():
    # Loading the data
    df = pd.read_csv('car_purchasing.csv')
    
    # Encoding the categorical features
    encoded_df = encode_categorical_features(df)
    
    # Define features and target variables
    X, y = define_features_target_variables(encoded_df)
    
    # Splitting the data into test and train sets
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Scale the data
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    # Create model instance
    models = initialize_model()

    # Train and predict
    predictions = train_and_predict(models, X_train_scaled, X_test_scaled, y_train)

    # Evaluate the model
    best_model, metrics_df = evaluate_model_performance(models, predictions, y_test)

    # Save the best model and scaler
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    print("Metrics DataFrame:\n")
    print(metrics_df)


# Call the main function
if __name__ == '__main__':
    main()
