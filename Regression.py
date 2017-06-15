import os
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score


def linear_regression(total_df_train_x, total_df_train_y, total_df_test_x, total_df_test_y):
    selection_list = ['elevation (m)', 'planar distance (m)', 'adjusted distance (m)', 'speed (m/s)', 'acceleration(m/s^2)']
    while(True):
        vif = pd.DataFrame()
        vif["VIF Factor"] = [variance_inflation_factor(total_df_train_x[selection_list].values, i) for i in range(total_df_train_x[selection_list].shape[1])]
        if len(selection_list) == 1 or vif['VIF Factor'].max() < 10:
            break
        else:
            del selection_list[int(vif.idxmax())]

    poly_linear = PolynomialFeatures(3, interaction_only = True, include_bias = True)
    truncated_transformed_total_df_train_x = poly_linear.fit_transform(total_df_train_x[selection_list])

    linreg = LinearRegression(normalize=True)
    linreg.fit(truncated_transformed_total_df_train_x, total_df_train_y)

    truncated_transformed_total_df_test_x = poly_linear.fit_transform(total_df_test_x[selection_list])
    y_predicted = linreg.predict(truncated_transformed_total_df_test_x)
    result = sum((y_predicted - total_df_test_y)**2)
    r_squared = r2_score(total_df_test_y, y_predicted)
    print(result, r_squared)


def linear_regression_l1_regularized(total_df_train_x, total_df_train_y, total_df_test_x, total_df_test_y):
    poly_lasso = PolynomialFeatures(3, interaction_only = False, include_bias = True)
    transformed_total_df_train_x = poly_lasso.fit_transform(total_df_train_x)
    transformed_total_df_test_x = poly_lasso.fit_transform(total_df_test_x)

    lassoreg = Lasso(alpha = 0.0001, normalize = True)
    lassoreg.fit(transformed_total_df_train_x, total_df_train_y)
    y_predicted = lassoreg.predict(transformed_total_df_test_x)

    result = sum((y_predicted - total_df_test_y)**2)
    r_squared = r2_score(total_df_test_y, y_predicted)
    print(result, r_squared)


def linear_regression_l2_regularized(total_df_train_x, total_df_train_y, total_df_test_x, total_df_test_y):
    poly_ridge = PolynomialFeatures(3, interaction_only = False, include_bias = True)
    transformed_total_df_train_x = poly_ridge.fit_transform(total_df_train_x)
    transformed_total_df_test_x = poly_ridge.fit_transform(total_df_test_x)

    ridgereg = Ridge(alpha = 0.0001, normalize = True)
    ridgereg.fit(transformed_total_df_train_x, total_df_train_y)
    y_predicted = ridgereg.predict(transformed_total_df_test_x)

    result = sum((y_predicted - total_df_test_y)**2)
    r_squared = r2_score(total_df_test_y, y_predicted)
    print(result, r_squared)


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

    linear_regression(total_df_train_x, total_df_train_y, total_df_test_x, total_df_test_y)
    linear_regression_l1_regularized(total_df_train_x, total_df_train_y, total_df_test_x, total_df_test_y)
    linear_regression_l2_regularized(total_df_train_x, total_df_train_y, total_df_test_x, total_df_test_y)


if __name__ == "__main__":
    main()
