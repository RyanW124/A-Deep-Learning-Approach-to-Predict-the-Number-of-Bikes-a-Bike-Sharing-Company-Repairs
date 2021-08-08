import pickle
import numpy as np
import math
from sklearn.cluster import KMeans


def latlong_to_distance(lat, long):
    return lat, math.cos(math.radians(lat)) * long

def cluster(station_file):
    with open(station_file, 'rb') as f:
        x, y = pickle.load(f)
    points = [latlong_to_distance(i[0], i[1]) for i in y]
    groups = KMeans(n_clusters=10, random_state=0).fit_predict(points)
    return x, y, groups, points