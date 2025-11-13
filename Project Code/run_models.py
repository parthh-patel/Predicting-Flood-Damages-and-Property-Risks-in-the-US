import pandas as pd
import numpy as np
from clean_data import run_clean_data
from utils import load_and_split_data
from datetime import datetime
import time
import os
from typing import List, Dict

import linear_regression_model as lr
import knn_model as knn
import random_forest as rf
import boosting as boosting
import joblib

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(text: str):
    clear_screen()
    print("=" * 75)
    print(f"{text:^75}")
    print("=" * 75)
    print()

def print_enter():
    print()
    print(f"{'Press Enter to continue...':^75}")
    input()

def print_menu(options: Dict[int, str]) -> None:
    for key, value in options.items():
        print(f"  [{key}] {value}")
    print()

def get_valid_input(prompt: str, valid_options: List[int]) -> int:
    while True:
        try:
            choice = int(input(prompt))
            if choice in valid_options:
                return choice
            print(f"Please enter a valid option ({min(valid_options)}-{max(valid_options)})")
        except ValueError:
            print("Please enter a valid number")

def load_pretrained_model():
    try:
        print("\nLoading pretrained model...")
        model = joblib.load("boosting_model.pkl")
        print("Model loaded successfully!")
        return model
    except FileNotFoundError:
        print("\nError: No pretrained model found! Please train the model first.")
        return None

def make_custom_predictions(model):
    try:
        custom_prediction_data = pd.read_csv("custom_predictions.csv")
        all_predictions = []

        for _, row in custom_prediction_data.iterrows():
            row_df = pd.DataFrame([row])
            custom_prediction = model.predict(row_df)
            row_with_prediction = row.copy()
            row_with_prediction['Predicted_Damage'] = custom_prediction[0]
            all_predictions.append(row_with_prediction)

        if all_predictions:
            custom_results = pd.DataFrame(all_predictions)
            custom_results.to_csv("custom_predictions_results.csv", index=False)
            print("\nCustom predictions have been saved to 'custom_predictions_results.csv'")
        else:
            print("No successful predictions could be made.")
    except Exception as e:
        print(f"Error making custom predictions: {str(e)}")

def train_models(X_train, X_test, y_train, y_test, seed):
    print_header("Model Selection")
    print("Available models:")
    
    models = {
        0: "Run ALL Models",
        1: "Linear Regression",
        2: "K-Nearest Neighbors",
        3: "Random Forest",
        4: "Boosting"
    }
    print_menu(models)
    
    selected_model = get_valid_input("Select a model (0-4): ", range(5))
    
    results_file = "model_results.csv"
    all_results = pd.DataFrame()

    models_to_run = range(1, 5) if selected_model == 0 else [selected_model]

    print_header("Training Models")
    
    for model_num in models_to_run:
        print(f"\nTraining {models[model_num]}...")
        print("-" * 30)
        
        try:
            start_time = time.time()
            
            if model_num == 1:
                print("Starting Linear Regression training...")
                results = lr.run_model(X_train, X_test, y_train, y_test)
            elif model_num == 2:
                print("Starting KNN training...")
                results = knn.run_model(X_train, X_test, y_train, y_test)
            elif model_num == 3:
                print("Starting Random Forest training...")
                results = rf.run_model(X_train, X_test, y_train, y_test)
            elif model_num == 4:
                print("Starting Boosting training...")
                results = boosting.run_model(X_train, X_test, y_train, y_test)
            
            training_time = time.time() - start_time
            
            results['Seed'] = seed
            results['Training Time (s)'] = training_time
            results['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            all_results = pd.concat([all_results, results], ignore_index=True)
            print(f"\n{models[model_num]} training completed!")
            print(f"Training time: {training_time:.2f} seconds")
            print("...\n" * 5)
            
        except Exception as e:
            print(f"Error training {models[model_num]}: {str(e)}")
            print_enter()
            continue

    all_results.to_csv(results_file, index=False)
    print_header("Training Results")
    print("Final Results:")
    print("\n" + all_results.to_string(index=False))
    print(f"\nResults have been saved to {results_file}")

def main():
    # main intro screen
    print_header("Predicting Flood Damages and Property Risks in the US")
                 
    print("Team 152: Allen Yen Chen, Yi-Chun Chen, Kevin Hsu, Parth Patel, David Harold Thompson \n\nML Model Training Interface")
    print("This program will help you train and evaluate various ML models")

    print_enter()

    # initial choice: Train or Predict
    print_header("Main Menu")
    print("Please select an option:")
    initial_options = {
        1: "Train new models",
        2: "Use pretrained model for predictions"
    }
    print_menu(initial_options)
    initial_choice = get_valid_input("Enter your choice (1-2): ", [1, 2])

    try:
        if initial_choice == 2:
            print_header("Custom Predictions")
            model = load_pretrained_model()
            if model:
                make_custom_predictions(model)
            return

        print_header("Data Preparation")
        print("Would you like to:")
        data_options = {
            1: "Clean raw data from FimaNfipClaims.csv",
            2: "Use existing cleaned data"
        }
        print_menu(data_options)
        data_choice = get_valid_input("Enter your choice (1-2): ", [1, 2])

        if data_choice == 1:
            print("\nLoading raw data...")
            df = pd.read_csv("FimaNfipClaims.csv")
            print("Cleaning data...")
            run_clean_data(df)
            print("\nData cleaned and saved as 'cleaned_data.csv'")
            time.sleep(2)
        else:
            if not os.path.exists("cleaned_data.csv"):
                print("\nError: cleaned_data.csv not found!")
                print("Please clean the raw data first.")
                return
            print("\nUsing existing cleaned dataset...")
            time.sleep(1)

        # load and split data
        print_header("Data Loading")
        print("Loading and splitting data...")
        X_train, X_test, y_train, y_test = load_and_split_data("cleaned_data.csv")
        print("\nData loaded successfully!")
        
        seed = get_valid_input("\nEnter a random seed for reproducibility: ", range(1, 1000001))
        np.random.seed(seed)

        train_models(X_train, X_test, y_train, y_test, seed)

    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
    
    finally:
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()