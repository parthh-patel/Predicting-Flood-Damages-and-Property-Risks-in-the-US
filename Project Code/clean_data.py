import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

def run_clean_data(df):
    df_clean = df.copy()
    
    # drop unnecessary columns
    columns_to_drop = [
        'lowestAdjacentGrade', 'floodCharacteristicsIndicator', 'agricultureStructureIndicator',
        'policyCount', 'houseWorship', 'nonProfitIndicator', 'smallBusinessIndicatorBuilding',
        'yearOfLoss', 'stateOwnedIndicator', 'rentalPropertyIndicator', 'reportedCity', 'id',
        'nfipRatedCommunityNumber', 'originalNBDate', 'countyCode', 'censusTract',
        'censusBlockGroupFips', 'ratedFloodZone', 'replacementCostBasis',
        'disasterAssistanceCoverageRequired', 'amountPaidOnBuildingClaim',
        'amountPaidOnContentsClaim', 'amountPaidOnIncreasedCostOfComplianceClaim',
        'floodEvent', 'ficoNumber', 'nonPaymentReasonContents', 'nfipCommunityName',
        'nfipCommunityNumberCurrent', 'nonPaymentReasonBuilding', 'crsClassificationCode',
        'eventDesignationNumber', 'buildingDeductibleCode', 'contentsDeductibleCode',
        'asOfDate', 'contentsReplacementCost', 'reportedZipCode',
        'netBuildingPaymentAmount', 'netContentsPaymentAmount', 'netIccPaymentAmount'
    ]
    df_clean = df_clean.drop(columns=columns_to_drop, errors='ignore')

    # Keep only primary residences
    df_clean = df_clean[df_clean['primaryResidenceIndicator'] == 1]
    
    df_clean['month'] = pd.to_datetime(df_clean['dateOfLoss'], errors='coerce').dt.month
    df_clean['year'] = pd.to_datetime(df_clean['dateOfLoss'], errors='coerce').dt.year
    
    df_clean['constructionYear'] = pd.to_datetime(df_clean['originalConstructionDate'], errors='coerce').dt.year

    # fill missing values here with the median of the values in the state
    df_clean['constructionYear'] = df_clean.groupby('state')['constructionYear'].transform(lambda x: x.fillna(x.median()))

    # convert categorical columns to strings and encode
    le = LabelEncoder()
    categorical_columns = [
        'state', 'condominiumCoverageTypeCode', 'rateMethod', 'causeOfDamage', 
        'elevationCertificateIndicator', 'floodZoneCurrent'
    ]
    
    mappings = {}
    for col in categorical_columns:
        df_clean[col] = df_clean[col].astype(str)
        
        df_clean[col] = df_clean[col].apply(lambda x: 
            str(float(x)) if x.replace('.','').isdigit() else x
        )
        
        df_clean[col] = df_clean[col].replace(['nan', 'None'], 'nan')
        
        df_clean[col] = le.fit_transform(df_clean[col])
        mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    # save mappings to a text file
    with open('categorical_mappings.txt', 'w') as f:
        for col, mapping in mappings.items():
            f.write(f"{col}:\n")
            sorted_items = sorted(mapping.items(), key=lambda x: x[1])
            for key, value in sorted_items:
                f.write(f"  {key}: {value}\n")
            f.write("\n")
    
    # group flood zones
    flood_zone_groups = {
        'A': 'A-ZONE', 'AE': 'A-ZONE', 'A1-A30': 'A-ZONE', 'AH': 'A-ZONE',
        'AO': 'A-ZONE', 'V': 'V-ZONE', 'VE': 'V-ZONE', 'V1-V30': 'V-ZONE',
        'X': 'X-ZONE', 'B': 'X-ZONE', 'C': 'X-ZONE', 'D': 'D-ZONE'
    }
    df_clean['floodZoneGroup'] = df_clean['floodZoneCurrent'].map(
        lambda x: next((v for k, v in flood_zone_groups.items() if str(x).startswith(k)), 'OTHER')
    )
    df_clean['floodZoneGroup'] = le.fit_transform(df_clean['floodZoneGroup'])
    
    # convert numeric columns and fill nans
    numeric_columns = [
        'totalBuildingInsuranceCoverage', 'totalContentsInsuranceCoverage', 'occupancyType',
        'numberOfUnits', 'numberOfFloorsInTheInsuredBuilding', 'latitude', 'longitude',
        'floodWaterDuration', 'waterDepth', 'buildingDamageAmount', 'buildingPropertyValue',
        'buildingReplacementCost', 'locationOfContents', 'iccCoverage', 'obstructionType',
        'contentsDamageAmount', 'contentsPropertyValue', 'buildingDescriptionCode',
        'basementEnclosureCrawlspaceType', 'elevationDifference', 'baseFloodElevation',
        'lowestFloorElevation'
    ]
    for col in numeric_columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    

    # drop original date columns
    df_clean = df_clean.drop(['dateOfLoss', 'originalConstructionDate'], axis=1)
    
    df_clean = df_clean.dropna(subset=['constructionYear'])

    df_clean.to_csv('cleaned_data.csv', index=False)

    return df_clean
