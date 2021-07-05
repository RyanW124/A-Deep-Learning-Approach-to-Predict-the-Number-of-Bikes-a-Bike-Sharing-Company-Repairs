import csv, numpy as np, datetime
from collections import defaultdict
from tqdm.notebook import tqdm_notebook
from os import listdir
import pickle
from dateutil.relativedelta import relativedelta

def save_stations(station_file):
    stations = {}
    for file in tqdm_notebook(listdir('../Data/Bike/')):
        with open('../Data/Bike/' + file) as f:
            reader = csv.reader(f)
            next(reader)
            for trip in reader:
                if trip[3] == 'NULL':
                    continue
                try:
                    if 40.5 < float(trip[5]) < 45:
                        stations[int(trip[3])] = [float(trip[5]), float(trip[6])]
                except ValueError:
                    continue
    x, y = [], []
    for station in stations:
        x.append(station)
        y.append(stations[station])
    with open(station_file, 'wb') as f:
        pickle.dump([x, y], f)

def load(groups, d_file):
    data = []
    date = datetime.date(month=6, year=2013, day=1)
    months = []
    for file in tqdm_notebook(listdir('../Data/Bike/')):
        data_point = [0] * 12
        with open('../Data/Bike/' + file) as f:
            reader = csv.reader(f)
            next(reader)
            n_trips = 0
            duration = 0
            data_point[0] = date.year
            data_point[1] = date.month
            for trip in reader:
                if trip[3] == 'NULL':
                    continue
                if int(trip[3]) not in groups:
                    continue
                data_point[groups[int(trip[3])] + 2] += 1
                duration += int(trip[0])
                n_trips += 1
            data_point.append(duration/n_trips)
            data_point.append(n_trips)
        date += relativedelta(months=1)
        data.append(data_point)
        months.append(data_point[2:12])
    data = data[:-1]
    with open('../Data/weather.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for index, weather in enumerate(reader):
            for i in range(1, 4):
                data[index].append(int(weather[i]))
    y1 = []
    y2 = []
    with open('../Data/repair.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for index, r in enumerate(reader):
            y1.append([int(r[1])])
            m = months[index+1]
            sum_m = sum(m)
            y2.append([i/sum_m for i in m])
    if d_file is not None:
        with open(d_file, 'wb') as f:
            pickle.dump([data, y1, y2], f)
    return data, y1, y2