import pandas as pd
import numpy as np

gt_file = "data/daily_county_data/25001_daily_data.csv"
pred_file = "results/COVID/25001/predictions.csv"

gt = pd.read_csv(gt_file)
pred = pd.read_csv(pred_file)

y = gt['deaths'].values
yhat = pred['predicted_deaths'].values

ND = np.sum(np.abs(y - yhat)) / np.sum(np.abs(y))
RMSE = np.sqrt(np.mean((y - yhat)**2))
MAE = np.mean(np.abs(y - yhat))

print("ND:", ND)
print("RMSE:", RMSE)
print("MAE:", MAE)
