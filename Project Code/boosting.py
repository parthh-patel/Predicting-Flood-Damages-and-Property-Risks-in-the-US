import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from utils import evaluate_model, save_metrics 
import lightgbm as lgb
from sklearn.pipeline import Pipeline
import joblib


def run_model(X_train, X_test, y_train, y_test, n_estimators_values=[100]):
    
    numeric_cols = [
        'elevatedBuildingIndicator', 'elevationDifference', 'baseFloodElevation',
        'lowestFloorElevation', 'postFIRMConstructionIndicator', 'totalBuildingInsuranceCoverage',
        'totalContentsInsuranceCoverage', 'primaryResidenceIndicator',
        'buildingPropertyValue', 'contentsDamageAmount',
        'contentsPropertyValue', 'floodWaterDuration', 'floodproofedIndicator', 'iccCoverage',
        'numberOfUnits', 'buildingReplacementCost', 'waterDepth',
        'latitude', 'longitude', 'year', 'constructionYear'
        ]
    categorical_cols = [
        'basementEnclosureCrawlspaceType', 'elevationCertificateIndicator', 'locationOfContents',
        'numberOfFloorsInTheInsuredBuilding', 'obstructionType', 'occupancyType', 'rateMethod',
        'causeOfDamage', 'condominiumCoverageTypeCode', 'floodZoneCurrent', 'buildingDescriptionCode',
        'state', 'month', 'floodZoneGroup'
        ]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='drop'
    )

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', lgb.LGBMRegressor(
            objective='regression',
            metric='mse',
            boosting_type='gbdt',
            num_leaves=300,
            max_depth=-1,
            learning_rate=0.01,
            n_estimators=400,
            bagging_fraction=0.8,
            feature_fraction=0.8,
            min_child_weight=1,
            lambda_l1=0.1,
            lambda_l2=0.1
            # device='gpu',
            # gpu_device_id=0,
        ))
    ])


    results_list = []
    
    for n_estimators in n_estimators_values:
        # print(f"\nTraining Random Forest model with n_estimators={n_estimators}")
        

        """
        model = lgb.LGBMRegressor()
        param_grid = {
            'num_leaves': [200],         # Example values for num_leaves
            'learning_rate': [0.01, 0.05, 0.1], # Example values for learning_rate
            'n_estimators': [300],         # Example values for n_estimators
            'lambda_l1': [0, 0.1, 1],           # Example values for L1 regularization
            'lambda_l2': [0, 0.1, 1],  
            'device': ['gpu'],
            #'early_stopping_rounds': [50],     
            'num_threads': [-1],               
            'gpu_device_id': [0], 
        }
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='neg_mean_squared_error',  # Use appropriate scoring metric for regression
            cv=2,                               # 5-fold cross-validation
            n_jobs=-1,                          # Use all available cores
            verbose=1                           # Print progress messages
        )
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        print(f"Best parameters: {best_params}")
        """
        model.named_steps['regressor'].n_estimators = n_estimators

        model.fit(X_train, y_train)

        joblib.dump(model, 'boosting_model.pkl')


        test_preds = model.predict(X_test)

        # test_results = pd.DataFrame({
        #     # 'state': X_test['state'] if 'state' in X_test.columns else 'Unknown',
        #     # 'property value': X_test['buildingPropertyValue'],
        #     # 'actual': y_test,

        #     'predicted': test_preds
        # })
        
        test_results = X_test
        test_results['preds'] = test_preds

        test_results.to_csv('test_set_preds.csv', index=False)

        # evaluate the model
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        
        results_list.append({
            'n_estimators': n_estimators,
            **metrics
        })
    
        # custom_prediction_data = pd.read_csv('custom_predictions.csv')
        

        # all_predictions = []
        # for _, row in custom_prediction_data.iterrows():
        #     row_df = pd.DataFrame([row])
            
        #     try:
        #         custom_prediction = model.predict(row_df)
                
        #         row_with_prediction = row.copy()
        #         row_with_prediction['Predicted_Damage'] = custom_prediction[0]
                
        #         all_predictions.append(row_with_prediction)
        #     except Exception as e:
        #         print(f"Error predicting for row: {row}")
        #         print(f"Error details: {e}")
        
        # if all_predictions:
        #     custom_results = pd.DataFrame(all_predictions)
        #     custom_results.to_csv('custom_predictions_results.csv', index=False)

        #     # print("\n\nCustom Predictions:")
        #     # print(custom_results[['state', 'constructionYear', 'Predicted_Damage']])
        # else:
        #     print("No successful predictions could be made.")


        results_df = save_metrics(results_list, 'Boosting', 'boosting.csv')

    return results_df

if __name__ == "__main__":
    pass
