import pandas as pd
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

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

    lstmUnits = 48
    numDimensions = 15

    tf.reset_default_graph()
    labels = tf.placeholder(tf.float64, [1, 1])
    input_data = tf.placeholder(tf.float64, [1, 6, numDimensions])
    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.85)
    value, _ = tf.nn.dynamic_rnn(lstmCell, input_data, dtype=tf.float64)
    weight = tf.Variable(tf.truncated_normal([lstmUnits, 1],dtype=tf.float64))
    bias = tf.Variable(tf.constant(0.1, shape=[1],dtype=tf.float64))
    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)
    error=tf.pow(tf.subtract(prediction,labels),2)
    train_step = tf.train.AdamOptimizer(0.0006).minimize(error)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(5,total_df_train_x.shape[0]):
        sess.run(train_step, feed_dict = {input_data: total_df_train_x.values[(i-5):(i+1)].reshape((1,6,15)), labels: total_df_train_y.values[i:(i+1)].reshape(1,1)})

    output_store = []

    for i in range(5,total_df_test_x.shape[0]):
        output_store.append(sess.run(prediction, feed_dict = {input_data: total_df_test_x.values[(i-5):(i+1)].reshape((1,6,15))})[0][0])

    print(mean_squared_error(output_store, total_df_test_y[5:]))
    print(r2_score(output_store, total_df_test_y[5:]))
    print(mean_absolute_error(output_store, total_df_test_y[5:]))


if __name__ == "__main__":
    main()
