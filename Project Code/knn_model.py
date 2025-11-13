import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from utils import evaluate_model, save_metrics


def run_model(X_train, X_test, y_train, y_test, k_values=[5, 10, 15]):

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results_list = []
    model_names = []

    for k in k_values:
        print(f"\nTraining KNN model with k={k}")
        knn_model = KNeighborsRegressor(n_neighbors=k)
        knn_model.fit(X_train_scaled, y_train)
        
        metrics = evaluate_model(knn_model, X_train_scaled, X_test_scaled, y_train, y_test)
        results_list.append(metrics)
        
        model_names.append(f"KNN (k={k})")
        print(f"Completed training and evaluation for k={k}")
    
    results_df = save_metrics(results_list, 'KNN', 'knn_results.csv', model_names=model_names)
    return results_df

if __name__ == "__main__":
    pass