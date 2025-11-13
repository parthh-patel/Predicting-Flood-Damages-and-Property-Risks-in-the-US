import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from utils import evaluate_model, save_metrics

def run_model(X_train, X_test, y_train, y_test):

    lr_model = LinearRegression()
    
    lr_model.fit(X_train, y_train)
    
    metrics = evaluate_model(lr_model, X_train, X_test, y_train, y_test)

    results_df = save_metrics(metrics, 'Linear Regression', 'linear_regression_results.csv')
    
    return results_df

if __name__ == "__main__":
    pass