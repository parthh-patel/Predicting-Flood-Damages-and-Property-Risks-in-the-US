import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def load_and_split_data(file_path, train_ratio=0.9):

    df_clean = pd.read_csv(file_path)
    df_clean = df_clean.sort_values(by=['year', 'month'])
    
    train_size = int(len(df_clean) * train_ratio)
    train_data = df_clean.iloc[:train_size]
    test_data = df_clean.iloc[train_size:]
    
    X_train = train_data.drop(columns=['buildingDamageAmount'])
    y_train = train_data['buildingDamageAmount']
    X_test = test_data.drop(columns=['buildingDamageAmount'])
    y_test = test_data['buildingDamageAmount']

    print("\nNumber of total data points:", len(df_clean))
    print("Training data range:")
    print(f"\tStarting: {int(X_train.iloc[0]['year'])}-{int(X_train.iloc[0]['month'])}, "
          f"\tEnding: {int(X_train.iloc[-1]['year'])}-{int(X_train.iloc[-1]['month'])}, "
          f'\tNumber of data points:', len(X_train))
    print("\nTesting data range:")
    print(f"\tStarting: {int(X_test.iloc[0]['year'])}-{int(X_test.iloc[0]['month'])}, "
          f"\tEnding: {int(X_test.iloc[-1]['year'])}-{int(X_test.iloc[-1]['month'])}, "
          f'\tNumber of data points:', len(X_test))
    
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_train, X_test, y_train, y_test):

    # make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # calc metrics
    metrics = {
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred)
    }
    
    return metrics

def save_metrics(metrics, model_name, output_file='model_evaluation_results.csv', model_names=None):
    """
    Save metrics to CSV file and return as DataFrame.
    For KNN, metrics can be either a single dict or a list of dicts.
    """
    if isinstance(metrics, list):
        results_list = []
        for i, metric in enumerate(metrics):
            result = {
                'Model': model_names[i] if model_names else f"{model_name} (Model {i+1})",
                'Train R²': metric['train_r2'],
                'Test R²': metric['test_r2'],
                'Train RMSE': metric['train_rmse'],
                'Test RMSE': metric['test_rmse'],
                'Train MAE': metric['train_mae'],
                'Test MAE': metric['test_mae']
            }
            results_list.append(result)
        results_df = pd.DataFrame(results_list)
    else: 
        results_df = pd.DataFrame({
            'Model': [model_name],
            'Train R²': [metrics['train_r2']],
            'Test R²': [metrics['test_r2']],
            'Train RMSE': [metrics['train_rmse']],
            'Test RMSE': [metrics['test_rmse']],
            'Train MAE': [metrics['train_mae']],
            'Test MAE': [metrics['test_mae']]
        })

    results_df.to_csv(output_file, index=False)
    print(f"\nResults have been saved to '{output_file}'")
    print("\nModel Evaluation Results:")
    print(results_df.to_string(index=False))
    
    return results_df