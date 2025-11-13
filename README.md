# Predicting Flood Damages and Property Risks in the US

**In this project, the relationship between property characteristics and potential flood damage was investigated. This was accomplished by comparing the robustness of four regression models: Boosting, Random Forest, Linear Regression, and K-Nearest Neighbors**

<div align="center">

| [Final Report](https://github.com/parthh-patel/Predicting-Flood-Damages-and-Property-Risks-in-the-US/blob/main/team152report.pdf) | [Project Code](https://github.com/parthh-patel/Predicting-Flood-Damages-and-Property-Risks-in-the-US/tree/main/Project%20Code) |
|---|---|

</div>

___

![](team152poster.png)

This project involved sourcing the dataset from FEMA's National Flood Insurance Program (NFIP), carrying out exploratory data analysis, fixing missing values and outliers, and fitting various regression models to predict flood damage amounts.

**Key Findings:**

- As evident from the table below, Boosting was the most effective predictive model for understanding the relationship between flood damage and the given features in this dataset. Random Forest performed comparably well with similar R² and error metrics, whereas Linear Regression and KNN performed poorly with negative R² values.

<div align="center">

| Ranking | Model             | R² (Test) | RMSE (Test) | MAE (Test) |
|---------|-------------------|-----------|-------------|------------|
| 1       | Boosting          | 0.32      | 42,571      | 13,316     |
| 2       | Random Forest     | 0.30      | 39,293      | 15,826     |
| 3       | Linear Regression | -67.01    | 414,503     | 50,810     |
| 4       | KNN (k=10)        | -189.94   | 694,555     | 37,637     |
| 5       | KNN (k=15)        | -236.39   | 774,427     | 44,515     |
| 6       | KNN (k=5)         | -278.25   | 839,939     | 36,485     |

</div>

- Through the analysis, *Building Property Value*, *Longitude*, *Year of Loss*, *Latitude*, *Building Replacement Cost*, and *Construction Year* were the most significant features for predicting flood damage. These features consistently ranked high in importance across both Random Forest and Boosting models.

**Dataset:**

- Source: National Flood Insurance Program (NFIP) claims data from FEMA.gov (996MB)
- 1.2 million residential property claim transactions
- 73 features per property (reduced to 36 after feature selection)
- Coverage period: 1979 to present
- Training set: Records prior to September 2019
- Testing set: Records from September 2019 onward

**Methodology:**

- Performed correlation analysis and forward stepwise regression to identify 36 significant predictors
- Filled missing values using state-level medians to preserve regional trends
- Split data temporally (90-10 ratio) to evaluate model performance on recent data
- Evaluated models using R², RMSE, and MAE metrics[
- Created interactive Tableau visualization for state-level risk mapping

**Project Innovations:**

- Property-specific flood damage prediction allowing homeowners to input their property characteristics and receive personalized damage estimates
- Interactive Tableau map displaying average predicted damage at the state level with adjustable feature sliders for granular analysis
