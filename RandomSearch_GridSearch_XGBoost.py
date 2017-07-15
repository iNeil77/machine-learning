from xgboost.sklearn import XGBRegressor
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

def random_search_param(total_df_train_x, total_df_train_y, total_df_test_x, total_df_test_y):
    param_grid = {'booster': ['gbtree', 'dart'], 'reg_lambda': [0.5, 0.8, 1], 'subsample': [0.5, 0.7], 'max_depth': [2, 3, 5], 'colsample_bytree': [0.5, 0.8], 'reg_alpha': [0, 0.3, 0.5], 'n_estimators': [10, 15, 20]}
    n_iter_search = 20
    random_search = RandomizedSearchCV(XGBRegressor(), param_distributions = param_dist, n_iter = n_iter_search, n_jobs = 4)

    random_search.fit(total_df_train_x, total_df_train_y)
    pred = random_search.predict(total_df_test_x)

    print(mean_squared_error(pred, total_df_test_y))
    print(mean_absolute_error(pred, total_df_test_y))
    print(r2_score(pred, total_df_test_y))
    print(random_search.best_estimator_)


def grid_search_param(total_df_train_x, total_df_train_y, total_df_test_x, total_df_test_y):
    param_grid = {'booster': ['gbtree', 'dart'], 'reg_lambda': [0.5, 0.8, 1], 'subsample': [0.5, 0.7], 'max_depth': [2, 3, 5], 'colsample_bytree': [0.5, 0.8], 'reg_alpha': [0, 0.3, 0.5], 'n_estimators': [10, 15, 20]}
    grid_search = GridSearchCV(XGBRegressor(), param_distributions = param_dist, n_iter = n_iter_search, n_jobs = 4)

    grid_search.fit(total_df_train_x, total_df_train_y)
    pred = grid_search.predict(total_df_test_x)

    print(mean_squared_error(pred, total_df_test_y))
    print(mean_absolute_error(pred, total_df_test_y))
    print(r2_score(pred, total_df_test_y))
    print(grid_search.best_estimator_)


def main():
    total_df = pd.read_csv('combined.txt', header = None)
    total_df.columns = ['GMT time', 'relative time (s)', 'elevation (m)', 'planar distance (m)', 'adjusted distance (m)', 'speed (m/s)', 'acceleration(m/s^2)', 'power based on model (kW)', 'actual power (kW)', 'current (amps)', 'voltage (V)']
    total_df = total_df[total_df['actual power (kW)'] < 0.0]
    total_df = total_df.dropna()

    total_df = total_df.sample(frac = 1).reset_index(drop = True)
    total_df_train_x = total_df[['elevation (m)', 'planar distance (m)', 'adjusted distance (m)', 'speed (m/s)', 'acceleration(m/s^2)']][0:69000]
    total_df_train_y = total_df['actual power (kW)'][0:69000]
    total_df_test_x = total_df[['elevation (m)', 'planar distance (m)', 'adjusted distance (m)', 'speed (m/s)', 'acceleration(m/s^2)']][69000:]
    total_df_test_y = total_df['actual power (kW)'][69000:]

    total_df = None

    total_df_train_x = total_df_train_x.reset_index(drop = True)
    total_df_train_y = total_df_train_y.reset_index(drop = True)
    total_df_test_x = total_df_test_x.reset_index(drop = True)
    total_df_test_y = total_df_test_y.reset_index(drop = True)

    total_df_train_x.shape
    total_df_train_y.shape
    total_df_test_x.shape
    total_df_test_y.shape

    random_search_param(total_df_train_x, total_df_train_y, total_df_test_x, total_df_test_y)
    grid_search_param(total_df_train_x, total_df_train_y, total_df_test_x, total_df_test_y)


if __name__ == "__main__":
    main()
