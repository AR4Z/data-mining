from numpy import genfromtxt
import numpy
import pandas as pd
from math import sqrt
import random
import operator

path_iris_csv = 'data/IRIS.csv'
attrs = pd.read_csv(path_iris_csv, usecols = ['sepal_length','sepal_width', 'petal_length', 'petal_width']).to_numpy()
classification = pd.read_csv(path_iris_csv, usecols=['species']).to_numpy()

adjacent_mtx = numpy.zeros(shape=(150, 150))

def fourth_euclidean_distance(attrs1, attrs2):
    return ((attrs1[0] - attrs2[0])**4 + (attrs1[1] - attrs2[1])**4 +
            (attrs1[2] - attrs2[2])**4 + (attrs1[3] - attrs2[3])**4) ** (0.25)

for i_flower in range(150):
    for x_flower in range(i_flower + 1, 150):
        adjacent_mtx[x_flower][i_flower] = fourth_euclidean_distance(attrs[x_flower], attrs[i_flower])

def classify(k=4, input_flw_attrs=[6,2.5,4.3,1.9]):
    distances = []
    for x_flower in range(150):
        distances.append((fourth_euclidean_distance(attrs[x_flower], input_flw_attrs), x_flower,))
    distances.sort(key=lambda tup: tup[0])

    k_elements  = distances[:k]
    classes_count = {
        'Iris-setosa': 0,
        'Iris-versicolor': 0,
        'Iris-virginica': 0
    }

    for k_elm in k_elements:
        index = k_elm[1]
        classes_count[classification[index][0]] += 1
    
    vals = list(classes_count.values())

    desired_keys = []
    for key, value in classes_count.items():
        if vals.count(value) > 1:
            desired_keys.append(key)

    if len(desired_keys) > 1:
        print(random.choice(desired_keys))
    else:
        print(max(classes_count.items(), key=operator.itemgetter(1))[0])

    print(classes_count)


classify()
