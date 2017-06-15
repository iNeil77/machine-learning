import os
import pandas as pd
from sklearn.manifold import TSNE

def main():
    file_list = os.listdir('.')
    df_list = [pd.read_csv(file, header = None) for file in file_list]

    total_df = pd.concat(df_list)
    total_df.columns = ['GMT time', 'relative time (s)', 'elevation (m)', 'planar distance (m)', 'adjusted distance (m)', 'speed (m/s)', 'acceleration(m/s^2)', 'power based on model (kW)', 'actual power (kW)', 'current (amps)', 'voltage (V)']
    total_df = total_df[['elevation (m)', 'planar distance (m)', 'adjusted distance (m)', 'speed (m/s)', 'acceleration(m/s^2)']]
    model = TSNE(n_components = 2, perplexity = 40, early_exaggeration = 5, random_state = 0)
    out = model.fit_transform(total_df)
    plt.scatter(out[:,0], out[:,1])
    plt.show()


if __name__ == "__main__":
    main()
