import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Mock data similar to user's image
data = {
    'Date': ['2010-01-15', '2011-01-15', '2012-01-15', '2013-01-15'],
    'State': ['AP', 'AP', 'AP', 'AP'],
    'District': ['Vizag', 'Vizag', 'Vizag', 'Vizag'],
    'Consumption_MW': [100, 110, 125, 140],
    'Production_MW': [110, 115, 120, 125]
}
df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year

# Training logic simulation
annual_data = df.groupby('Year')[['Consumption_MW', 'Production_MW']].mean().reset_index()
print("Annual Data:\n", annual_data)

X = annual_data[['Year']]
y_cons = annual_data['Consumption_MW']

# Linear Regression
lr = LinearRegression()
lr.fit(X, y_cons)

# RF
rf = RandomForestRegressor(n_estimators=10, random_state=42)
rf.fit(X, y_cons)

# XGB
xgb_model = xgb.XGBRegressor(n_estimators=10, random_state=42)
xgb_model.fit(X, y_cons)

years_to_predict = [2026, 2027, 2028, 2029, 2030]
print("\nPredictions:")
for y in years_to_predict:
    test_X = np.array([[y]])
    p_lr = lr.predict(test_X)[0]
    p_rf = rf.predict(test_X)[0]
    p_xgb = xgb_model.predict(test_X)[0]
    print(f"Year {y}: LR={p_lr:.2f}, RF={p_rf:.2f}, XGB={p_xgb:.2f}")
