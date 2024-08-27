# ML-Regression-Model-Flight-Fare-Prediction
## Problem Statement
Create a regression model to predict flight fares using features like departure and arrival locations, flight duration, airline, and travel date. The aim is to accurately estimate flight prices, helping travelers with their planning and airlines with their pricing strategies.
## Data Preparation

### Handling Missing Values
Missing values in the dataset were handled using appropriate imputation techniques to ensure data completeness.

### Feature Engineering
New features were derived from existing columns, such as `Total_Stops` calculated from the `Route`, to enhance the model's predictive capability.

## Exploratory Data Analysis (EDA)
EDA was performed to understand the distribution of data, identify patterns, and uncover relationships between variables that impact flight fares.

## Preprocess Data

### Encoding Categorical Variables
Categorical variables such as `Airline`, `Source`, and `Destination` were encoded using one-hot encoding to convert them into a numerical format suitable for machine learning models.

### Split Data
The dataset was split into training and testing sets to evaluate the model's performance on unseen data.

## Correlation Analysis and VIF Calculation for Feature Selection
Correlation analysis and Variance Inflation Factor (VIF) were used to identify and remove multicollinearity, ensuring that the selected features contribute independently to the model.

## Model Building

Multiple machine learning models were built, including RandomForestRegressor, XGBRegressor, GradientBoostingRegressor, and KNeighborsRegressor, to predict flight fares.

### Observation
- **RandomForestRegressor**: High training R² of 94.68% and test R² of 78.88%, indicating potential overfitting.
- **DecisionTreeRegressor**: High training R² of 96.13% but lower test R² of 68.69%, showing clear overfitting.
- **XGBRegressor**: Balanced performance with a training R² of 90.67% and the highest test R² of 84.39%, making it the most reliable model.
- **GradientBoostingRegressor** and **KNeighborsRegressor** also showed decent test R² scores but were outperformed by XGBRegressor.

## Hyperparameter Tuning

Hyperparameter tuning was performed to optimize the models, with XGBRegressor showing the best generalization and robustness across different configurations.

## Heteroscedasticity Analysis

Residual analysis indicated the presence of heteroscedasticity, confirmed by the White test. The log transformation was applied to reduce heteroscedasticity, improving the model's performance.

## Final Results

1. **XGBRegressor**: Achieved an R² score of 0.848 after log transformation, with improved accuracy and reduced heteroscedasticity compared to the original model (training R² of 0.9105 and test R² of 0.8376).
2. The final model demonstrates strong predictive capability, with consistent scores across cross-validation folds, making it the most reliable for this problem.

### Final Observation
The XGBRegressor model demonstrates reliable and effective performance with a mean R² score of approximately 0.839, indicating strong predictive capability. The log transformation improved the model's fit and accuracy by reducing heteroscedasticity.

## Conclusion

The project successfully predicted flight fares with high accuracy using XGBRegressor, demonstrating its effectiveness in generalizing to unseen data. The model's robustness and accuracy make it a valuable tool for predicting flight fares in a real-world setting.
