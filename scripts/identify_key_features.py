import pandas as pd
from joblib import load
from sklearn.metrics import mean_squared_error, mean_absolute_error

test_features = load('../output/data/Data - Test Features.joblib').drop(columns=['StateName','CountyName','fips','date'])
model         = load('../output/models_predictions_nopca/RandomForestRegressor - Model 0 - 0.8 0.9.joblib')
importances   = model.feature_importances_
df = pd.DataFrame(importances)
df['name'] = test_features.columns
df = df.sort_values(by=0, ascending=False)
print(df.head(50))