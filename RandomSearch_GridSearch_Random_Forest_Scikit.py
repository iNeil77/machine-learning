import numpy as np
from time import time
from scipy.stats import randint as sp_randint
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

def random_search_param(total_df_train_x, total_df_train_y, total_df_test_x, total_df_test_y):
    rf = RandomForestRegressor(n_estimators=20)
    param_dist = {"max_depth": [3, None], "max_features": sp_randint(1, 5), "min_samples_split": sp_randint(2, 11), "min_samples_leaf": sp_randint(1, 11), "bootstrap": [True, False], "criterion": ["mae", "mse"]}

    n_iter_search = 20
    random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=n_iter_search)
    random_search.fit(total_df_train_x, total_df_train_y)
    pred = random_search.predict(total_df_test_x)
    print(mean_squared_error(pred, total_df_test_y))
    print(mean_absolute_error(pred, total_df_test_y))


def grid_search_param(total_df_train_x, total_df_train_y, total_df_test_x, total_df_test_y):
    rf = RandomForestRegressor(n_estimators=20)
    param_dist = {"max_depth": [3, None], "max_features": sp_randint(1, 5), "min_samples_split": sp_randint(2, 11), "min_samples_leaf": sp_randint(1, 11), "bootstrap": [True, False], "criterion": ["mae", "mse"]}

    grid_search = GridSearchCV(rf, param_grid=param_grid)
    grid_search.fit(total_df_train_x, total_df_train_y)
    pred = grid_search.predict(total_df_test_x)
    print(mean_squared_error(pred, total_df_test_y))
    print(mean_absolute_error(pred, total_df_test_y))


def main():
    file_list = os.listdir('.')
    df_list = [pd.read_csv(file, header = None) for file in file_list]

    total_df = pd.concat(df_list)
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
    grid_search_params(total_df_train_x, total_df_train_y, total_df_test_x, total_df_test_y)


if __name__ == "__main__":
    main()
