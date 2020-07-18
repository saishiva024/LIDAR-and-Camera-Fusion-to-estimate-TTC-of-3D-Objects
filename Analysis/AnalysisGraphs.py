import matplotlib.pyplot as plt
import csv
import numpy as np
import os
from os.path import join


class Graph:
    def __init__(self):
        self.lidar_ttc = list()
        self.camera_ttc = list()


graphs = dict()
counter = 1
numImages = 18

with open(r"../report/Report.csv", 'r') as f:
    data = csv.reader(f)
    next(data)
    for row in data:
        algo_combo = row[1] + "_" + row[2]

        if counter == 1:
            graphs[algo_combo] = Graph()

        graphs[algo_combo].camera_ttc.append(float(row[8]))
        graphs[algo_combo].lidar_ttc.append(float(row[9]))

        counter += 1

        if counter == numImages:
            counter = 1

x = list(np.arange(1, numImages))

for algo, graph_info in graphs.items():
    plt.plot(x, graph_info.lidar_ttc, label="LIDAR")
    plt.plot(x, graph_info.camera_ttc, label="CAMERA")
    plt.xlabel("Frame")
    plt.ylabel("TTC")
    plt.title(algo)
    plt.legend()
    plt.savefig(join(join(os.getcwd(), "Graphs"), algo+".png"))

    plt.clf()
    # plt.show()

print('ok')
