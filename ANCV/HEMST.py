import time

from EMST import generate_full_emst
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mat2csv import *
import pandas as pd


class HEMST:
    class RecHEMST:
        def __init__(self, n_clusters: int,
                     data,
                     depth=0):
            self.desired_k = n_clusters
            self.n_c = 1
            self.emst = None
            self.data = data
            self.labels = None
            self.weights = None
            self.data_type = None
            self.depth = depth

        def _pre_fit(self):
            data = self.data
            self.emst, self.weights = generate_full_emst(data)
            self.d = data.shape[1]
            self.data_type = type(data[0, 0])
            weights_values = list(self.weights.values())
            average_weight = np.average(weights_values)
            weights_std = np.std(weights_values)
            for e in self.emst.edges():
                if self.weights[e] > average_weight + weights_std:
                    self.n_c += 1
                    self.emst.remove_edge(e[0], e[1])
                    del self.weights[e]
            self.labels = np.empty((self.data.shape[0], 2), dtype=int)
            for i, component in enumerate(nx.connected_components(self.emst)):
                nodes = np.array(list(component))

                for n in nodes:
                    self.labels[n, 0] = n
                    self.labels[n, 1] = i

        def _remove_longest_edges(self):
            difference = self.desired_k - self.n_c
            longest_edges = sorted(self.weights.keys(), key=lambda x: self.weights[x], reverse=True)
            self.emst.remove_edges_from(longest_edges[:difference])
            self.n_c += difference
            for i, component in enumerate(nx.connected_components(self.emst)):
                nodes = np.array(list(component))
                for n in nodes:
                    self.labels[n, 1] = i
            assert self.n_c == self.desired_k
            assert self.n_c == nx.number_connected_components(self.emst)

        def _map_to_representants(self):
            while self.n_c > self.desired_k:
                components = [[self.labels[i, 0] for i in range(self.labels.shape[0]) if self.labels[i, 1] == j] for j
                              in np.unique(self.labels[:, 1])]
                current_mapping = {}

                S = np.empty(shape=(len(components), self.d), dtype=self.data_type)
                for i, c in enumerate(components):
                    nodes = c

                    current_data = self.data[nodes, :]
                    current_centroid = np.mean(current_data, axis=0)
                    centroid_id = np.argmin(np.linalg.norm(current_centroid - current_data, axis=1))

                    centroid = current_data[centroid_id, :]
                    S[i, :] = centroid
                    for n in nodes:
                        current_mapping[n] = i
                nh = HEMST.RecHEMST(n_clusters=self.desired_k,
                                    data=S, depth=self.depth + 1)
                nh.fit()
                new_labels = nh.get_labels()
                self.n_c = len(np.unique(new_labels[:, 1]))
                for i in range(self.labels.shape[0]):
                    self.labels[i, 1] = new_labels[current_mapping[i], 1]

        def fit(self):
            while self.n_c != self.desired_k:
                self._pre_fit()
                if self.n_c < self.desired_k:
                    self._remove_longest_edges()
                if self.n_c > self.desired_k:
                    self._map_to_representants()

        def get_labels(self):
            return self.labels

    def __init__(self, n_clusters=2):
        self.K = n_clusters

    def set_params(self, n_clusters=None):
        if n_clusters is not None:
            self.K = n_clusters

    def fit_predict(self, data):
        nh = self.RecHEMST(self.K, data)
        nh.fit()
        return nh.get_labels()[:, 1]


def main():
    datasets_list = ['basic1', 'basic2', 'basic3', 'basic4', 'basic5',
                     'blob', 'box', 'boxes', 'boxes2', 'boxes3', 'chrome', 'dart', 'dart2',
                     'face', 'lines', 'lines2', 'network', 'ring',
                     'spiral', 'spiral2', 'spirals', 'supernova', 'un', 'un2', 'wave']
    data = pd.read_csv(
        'D:\\02_code\\01_Python\\Internal-index\\HB-Acceleration\\datasets\\artificial\\'
        + datasets_list[0] + '.csv',
        header=None).values
    # 读取.mat
    # input_file_name = './datasets/smileface.mat'
    # data = read_file(input_file_name)
    st = time.time()
    # 数据标准化
    data_max = np.max(data, axis=0)
    data_min = np.min(data, axis=0)
    bre = []
    lk = 0
    for j in range(data.shape[1]):
        if data_max[j] - data_min[j] <= 0.0001:
            bre.append(j)
        else:
            data[:, j] = (data[:, j] - data_min[j]) / (data_max[j] - data_min[j])

    hemst = HEMST(n_clusters=4)
    labs = hemst.fit_predict(data)
    plt.scatter(data[:, 0], data[:, 1], c=labs)
    plt.show()
    print(labs)


if __name__ == '__main__':
    main()
