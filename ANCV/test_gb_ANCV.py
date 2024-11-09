from NTHC_Clustering import nthc
import networkx as nx
from validity_index import *
from HyperBallClustering_acceleration_v4 import *
import time


def get_all_centers(hb):
    centers = []
    for ball in hb:
        if len(ball) > 2:
            center = ball.mean(0)
            centers.append(center)
    return centers


def validity_index(data, labels, K, SNN_thr, Dataset_MST, Eu_dist):
    k_value = len(np.unique(labels))
    n, m = data.shape
    dist_index = np.argsort(Eu_dist, axis=1)
    knn = dist_index[:, 1:K + 1]  # 每个点的K近邻
    # Compute the Compactness
    com = Compactness_r9(data, labels, Dataset_MST, SNN_thr, knn, Eu_dist)
    Com = np.mean(com)

    # Compute the Separation
    sep_clu = Separation_r1(data, labels, Dataset_MST, knn, Eu_dist)
    index = np.zeros(k_value)
    for i in range(k_value):
        sep_c = sep_clu[i, np.where(sep_clu[i, :] != 0)[0]]
        index[i] = (np.max(sep_c) - com[i]) / max(np.max(sep_c), com[i])

    result = np.mean(index)

    sep = np.tril(sep_clu)
    sep = sep[sep > 0]
    Sep = np.mean(sep)
    return Sep, Com, sep_clu, result


def compute_mst(data):
    dist = squareform(pdist(data))
    xishu = nx.Graph(dist)  # create a graph from the distance matrix
    G = nx.minimum_spanning_tree(xishu)  # compute the minimum spanning tree
    T = nx.to_numpy_array(G)  # convert the MST to a numpy array
    return T


if __name__ == "__main__":

    datasets = ['n2', 'n3', 'face', 'D13', 'spirals', 'network', 'basic5', 'basic3', 'un2', 'boxes',
                'basic1', 'basic4']
    for i in range(datasets.__len__()):
        keys = [datasets[i]]
        st = time.time()
        data_path = "D:\\02_code\\01_Python\\Internal-index\\HB-Acceleration\\datasets\\data\\"
        hb_list_temp, data = hbc(keys, data_path)
        data = np.array(get_all_centers(hb_list_temp))

        SNN_thr = 3
        N, dim = data.shape
        K_max = int(np.ceil(np.sqrt(N)))
        k_value = 2
        score = []
        K = 10
        Eu_dist = squareform(pdist(data))
        mst = compute_mst(data)
        while k_value <= K_max:
            cluster = nthc
            clusters, labels = cluster(data, k_value, ka=10, la=0.2, si=0.65)
            sep, com, clu, result = validity_index(data, labels, K, SNN_thr, mst, Eu_dist)
            ancv = sep - com
            print(f"score: {ancv}")
            print('on:', k_value)
            score.append(ancv)
            k_value += 1
        on = np.argmax(score) + 2
        ed = time.time()
        t = ed - st
        print(f"dataset is: {datasets[i]}, 粒球优化ANCV经历了：{t:.2f}s, on is:{on}")
        print("------------------------------------------------")
