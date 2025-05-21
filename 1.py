import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from vecstack import stacking
df = pd.read_csv("train_data.csv")
target = df["target"]
train = df.drop("target")
X_train, X_test, y_train, y_test = train_test_split(
train, target, test_size=0.20)
model_1 = LinearRegression()
model_2 = xgb.XGBRegressor()
model_3 = RandomForestRegressor()
all_models = [model_1, model_2, model_3]
s_train, s_test = stacking(all_models, X_train, X_test,
y_train, regression=True, n_folds=4)
final_model = model_1
del = final_model.fit(s_train, y_train) pred_final = final_model.predict(X_test)
print(mean_squared_error(y_test, pred_final))
