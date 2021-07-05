import pickle
from sklearn import cluster as c


def cluster(station_file):
    with open(station_file, 'rb') as f:
        x, y = pickle.load(f)
    groups = c.KMeans(n_clusters=10, random_state=0).fit_predict(y)
    return x, y, groups