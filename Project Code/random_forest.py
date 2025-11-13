import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from utils import evaluate_model, save_metrics 

def run_model(X_train, X_test, y_train, y_test, n_estimators_values=[20]):

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
    
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    
    results_list = []
    
    for n_estimators in n_estimators_values:
        print(f"\nTraining Random Forest model with n_estimators={n_estimators}")
        
        rf_model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=42, 
            n_jobs=-1         
        )
        
        rf_model.fit(X_train, y_train)
        
        metrics = evaluate_model(rf_model, X_train, X_test, y_train, y_test)
        
        results_list.append({
            'n_estimators': n_estimators,
            **metrics
        })
    
    results_df = save_metrics(results_list, 'Random Forest', 'random_forest.csv')

    return results_df
