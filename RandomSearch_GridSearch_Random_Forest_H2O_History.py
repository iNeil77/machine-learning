import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.grid.grid_search import H2OGridSearch
import operator
import os
import pandas as pd

def random_search_param(train, validate, test):
    x = [2, 3, 4, 5, 6, 13, 14, 15, 16, 17, 24, 25, 26, 27, 28]
    y = 8

    hyper_parameters = {'min_split_improvement': [1e-07, 1e-05, 0.0001], 'max_depth': [20, 10], 'ntrees': [10, 30, 50], 'min_rows': [5, 15, 40], 'col_sample_rate_per_tree': [0.8, 0.5, 0.4]}
    search_criteria = { 'strategy': "RandomDiscrete", 'max_models': 3 }
    random_search = H2OGridSearch(H2ORandomForestEstimator, hyper_params = hyper_parameters, search_criteria = search_criteria)
    random_search.train(x = x, y = y, training_frame = train, validation_frame = validate, seed = 1234)

    show_list = ['col_sample_rate_per_tree', 'max_depth', 'min_rows', 'min_split_improvement', 'ntrees']
    r2_score = random_search.r2()
    mse_score = random_search.mse()
    model_data = random_search.sorted_metric_table()

    print(r2_score[max(r2_score.items(), key=operator.itemgetter(1))[0]])
    print(model_data[model_data['model_ids']==max(r2_score.items(), key=operator.itemgetter(1))[0]][show_list])
    print(mse_score[min(r2_score.items(), key=operator.itemgetter(1))[0]])
    print(model_data[model_data['model_ids']==min(mse_score.items(), key=operator.itemgetter(1))[0]][show_list])


def grid_search_param(train, validate, test):
    x = [2, 3, 4, 5, 6, 13, 14, 15, 16, 17, 24, 25, 26, 27, 28]
    y = 8

    hyper_parameters = {'min_split_improvement': [1e-07, 1e-05, 0.0001], 'max_depth': [20, 10], 'ntrees': [10, 30, 50], 'min_rows': [5, 15, 40], 'col_sample_rate_per_tree': [0.8, 0.5, 0.4]}
    search_criteria = {'strategy': "Cartesian"}
    grid_search = H2OGridSearch(H2ORandomForestEstimator, hyper_params = hyper_parameters, search_criteria = search_criteria)
    grid_search.train(x = x, y = y, training_frame = train, validation_frame = validate, seed = 1234)

    show_list = ['col_sample_rate_per_tree', 'max_depth', 'min_rows', 'min_split_improvement', 'ntrees']
    r2_score = grid_search.r2()
    mse_score = grid_search.mse()
    model_data = grid_search.sorted_metric_table()

    print(r2_score[max(r2_score.items(), key=operator.itemgetter(1))[0]])
    print(model_data[model_data['model_ids']==max(r2_score.items(), key=operator.itemgetter(1))[0]][show_list])
    print(mse_score[min(r2_score.items(), key=operator.itemgetter(1))[0]])
    print(model_data[model_data['model_ids']==min(mse_score.items(), key=operator.itemgetter(1))[0]][show_list])


def main():
    file_list = os.listdir('.')
    df_list = [pd.read_csv(file, header = None) for file in file_list]

    column_list = ['GMT time', 'relative time (s)', 'elevation (m)', 'planar distance (m)', 'adjusted distance (m)', 'speed (m/s)', 'acceleration(m/s^2)', 'power based on model (kW)', 'actual power (kW)', 'current (amps)', 'voltage (V)']
    column_list_del1 = ['GMT time Del1', 'relative time (s) Del1', 'elevation (m) Del1', 'planar distance (m) Del1', 'adjusted distance (m) Del1', 'speed (m/s) Del1', 'acceleration(m/s^2) Del1', 'power based on model (kW) Del1', 'actual power (kW) Del1', 'current (amps) Del1', 'voltage (V) Del1']
    column_list_del2 = ['GMT time Del2', 'relative time (s) Del2', 'elevation (m) Del2', 'planar distance (m) Del2', 'adjusted distance (m) Del2', 'speed (m/s) Del2', 'acceleration(m/s^2) Del2', 'power based on model (kW) Del2', 'actual power (kW) Del2', 'current (amps) Del2', 'voltage (V) Del2']

    train_temp = pd.DataFrame(columns = column_list + column_list_del1 + column_list_del2)
    validate_temp = pd.DataFrame(columns = column_list + column_list_del1 + column_list_del2)
    test_temp = pd.DataFrame(columns = column_list + column_list_del1 + column_list_del2)

    for df in df_list:
        del1_input = pd.DataFrame([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        del2_input = pd.DataFrame([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        del1_input = del1_input.append(df[1:df.shape[0]].values - df[0:(df.shape[0] - 1)],ignore_index=True)
        del2_input = del2_input.append(del1_input[2:del1_input.shape[0]].values - del1_input[1:(del1_input.shape[0] - 1)], ignore_index = True)

        df.columns = column_list
        del1_input.columns = column_list_del1
        del2_input.columns = column_list_del2

        total_input = pd.DataFrame(columns = column_list + column_list_del1 + column_list_del2)
        train_input = pd.DataFrame(columns = column_list + column_list_del1 + column_list_del2)
        validate_input = pd.DataFrame(columns = column_list + column_list_del1 + column_list_del2)
        test_input = pd.DataFrame(columns = column_list + column_list_del1 + column_list_del2)

        total_input = pd.concat([df, del1_input, del2_input], axis = 1)
        train_input = total_input[2:(int)(total_input.shape[0] * 0.85)]
        validate_input = total_input[(int)(total_input.shape[0] * 0.85):(int)(total_input.shape[0] * 0.95)]
        test_input = total_input[(int)(total_input.shape[0] * 0.95):]

        train_temp = pd.concat([train_temp, train_input], ignore_index = True)
        validate_temp = pd.concat([validate_temp, validate_input], ignore_index = True)
        test_temp = pd.concat([test_temp, test_input], ignore_index = True)


    train = h2o.H2OFrame(train_temp)
    validate = h2o.H2OFrame(validate_temp)
    test = h2o.H2OFrame(test_temp)

    random_search_param(train, validate, test)
    grid_search_param(train, validate, test)

    h2o.cluster().shutdown()


if __name__ == "__main__":
    h2o.init(nthreads=-1)
    main()
