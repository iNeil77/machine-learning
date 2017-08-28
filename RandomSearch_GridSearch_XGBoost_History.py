from xgboost.sklearn import XGBRegressor
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import os

def random_search_param(total_df_train_x, total_df_train_y, total_df_test_x, total_df_test_y):
    param_grid = {'booster': ['gbtree', 'dart'], 'reg_lambda': [0.5, 0.8, 1], 'subsample': [0.5, 0.7], 'max_depth': [2, 3, 5], 'colsample_bytree': [0.5, 0.8], 'reg_alpha': [0, 0.3, 0.5], 'n_estimators': [10, 15, 20]}
    n_iter_search = 20
    random_search = RandomizedSearchCV(XGBRegressor(), param_distributions = param_grid, n_iter = n_iter_search, n_jobs = 4)

    random_search.fit(total_df_train_x, total_df_train_y)
    pred = random_search.predict(total_df_test_x)

    print(mean_squared_error(pred, total_df_test_y))
    print(mean_absolute_error(pred, total_df_test_y))
    print(r2_score(pred, total_df_test_y))
    print(random_search.best_estimator_)


def grid_search_param(total_df_train_x, total_df_train_y, total_df_test_x, total_df_test_y):
    param_grid = {'booster': ['gbtree', 'dart'], 'reg_lambda': [0.5, 0.8, 1], 'subsample': [0.5, 0.7], 'max_depth': [2, 3, 5], 'colsample_bytree': [0.5, 0.8], 'reg_alpha': [0, 0.3, 0.5], 'n_estimators': [10, 15, 20]}
    grid_search = GridSearchCV(XGBRegressor(), param_grid = param_grid, n_jobs = 4)

    grid_search.fit(total_df_train_x, total_df_train_y)
    pred = grid_search.predict(total_df_test_x)

    print(mean_squared_error(pred, total_df_test_y))
    print(mean_absolute_error(pred, total_df_test_y))
    print(r2_score(pred, total_df_test_y))
    print(grid_search.best_estimator_)


def main():
    file_list = os.listdir('.')
    df_list = [pd.read_csv(file, header = None) for file in file_list]

    column_list = ['GMT time', 'relative time (s)', 'elevation (m)', 'planar distance (m)', 'adjusted distance (m)', 'speed (m/s)', 'acceleration(m/s^2)', 'power based on model (kW)', 'actual power (kW)', 'current (amps)', 'voltage (V)']
    column_list_del1 = ['GMT time Del1', 'relative time (s) Del1', 'elevation (m) Del1', 'planar distance (m) Del1', 'adjusted distance (m) Del1', 'speed (m/s) Del1', 'acceleration(m/s^2) Del1', 'power based on model (kW) Del1', 'actual power (kW) Del1', 'current (amps) Del1', 'voltage (V) Del1']
    column_list_del2 = ['GMT time Del2', 'relative time (s) Del2', 'elevation (m) Del2', 'planar distance (m) Del2', 'adjusted distance (m) Del2', 'speed (m/s) Del2', 'acceleration(m/s^2) Del2', 'power based on model (kW) Del2', 'actual power (kW) Del2', 'current (amps) Del2', 'voltage (V) Del2']

    input_temp = pd.DataFrame(columns = column_list + column_list_del1 + column_list_del2)

    for df in df_list:
        del1_input = pd.DataFrame([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        del2_input = pd.DataFrame([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        del1_input = del1_input.append(df[1:df.shape[0]].values - df[0:(df.shape[0] - 1)],ignore_index=True)
        del2_input = del2_input.append(del1_input[2:del1_input.shape[0]].values - del1_input[1:(del1_input.shape[0] - 1)], ignore_index = True)
        df.columns = column_list
        del1_input.columns = column_list_del1
        del2_input.columns = column_list_del2
        total_input = pd.DataFrame(columns = column_list + column_list_del1 + column_list_del2)
        total_input = pd.concat([df, del1_input, del2_input], axis = 1)
        input_temp = pd.concat([input_temp, total_input], ignore_index = True)

    total_df_train_x = input_temp[['elevation (m)', 'planar distance (m)', 'adjusted distance (m)', 'speed (m/s)', 'acceleration(m/s^2)', 'elevation (m) Del1', 'planar distance (m) Del1', 'adjusted distance (m) Del1', 'speed (m/s) Del1', 'acceleration(m/s^2) Del1','elevation (m) Del2', 'planar distance (m) Del2', 'adjusted distance (m) Del2', 'speed (m/s) Del2', 'acceleration(m/s^2) Del2']][2:(int)(input_temp.shape[0]*0.9)]
    total_df_train_y = input_temp['actual power (kW)'][2:(int)(input_temp.shape[0]*0.9)]
    total_df_test_x = input_temp[['elevation (m)', 'planar distance (m)', 'adjusted distance (m)', 'speed (m/s)', 'acceleration(m/s^2)', 'elevation (m) Del1', 'planar distance (m) Del1', 'adjusted distance (m) Del1', 'speed (m/s) Del1', 'acceleration(m/s^2) Del1','elevation (m) Del2', 'planar distance (m) Del2', 'adjusted distance (m) Del2', 'speed (m/s) Del2', 'acceleration(m/s^2) Del2']][(int)(input_temp.shape[0]*0.9):]
    total_df_test_y = input_temp['actual power (kW)'][(int)(input_temp.shape[0]*0.9):]

    total_df_train_x.dropna().shape
    total_df_train_y.dropna().shape
    total_df_test_x.dropna().shape
    total_df_test_y.dropna().shape

    random_search_param(total_df_train_x, total_df_train_y, total_df_test_x, total_df_test_y)
    grid_search_param(total_df_train_x, total_df_train_y, total_df_test_x, total_df_test_y)


if __name__ == "__main__":
    main()
