import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.grid.grid_search import H2OGridSearch
import operator

def random_search_param(train, validate, test):
    x = [2,3,4,5,6]
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
    x = [2,3,4,5,6]
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
    data = h2o.import_file("combined.txt")
    data = data[data[8] < 0]
    train, validate, test = data.split_frame([0.8, 0.15], seed=1234)

    random_search_param(train, validate, test)
    grid_search_param(train, validate, test)

    h2o.cluster().shutdown()


if __name__ == "__main__":
    h2o.init(nthreads=-1)
    main()
