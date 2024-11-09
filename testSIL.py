from HyperBallClustering_acceleration_v4 import *
import time
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score


def testSil(D):
    max_k = int(np.ceil(np.sqrt(D.shape[0])))
    k_list = range(2, max_k)

    best_silhouette = -1
    optimal_k = None
    silhouette_scores = []

    for k in k_list:
        Z = linkage(D, method='ward')
        labels = fcluster(Z, k, criterion='maxclust')

        if len(np.unique(labels)) > 1:
            silhouette_avg = silhouette_score(D, labels, metric='euclidean')
            silhouette_scores.append(silhouette_avg)

            if silhouette_avg > best_silhouette:
                best_silhouette = silhouette_avg
                optimal_k = k
        else:
            silhouette_scores.append(-1)

    on = optimal_k
    onCVI = best_silhouette
    eva = silhouette_scores

    return on, onCVI, eva


def get_all_centers(hb):
    centers = []
    for ball in hb:
        if len(ball) > 2:
            center = ball.mean(0)
            centers.append(center)
    return centers


def main():
    datasets = ['n2', 'n3', 'face', 'D13', 'spirals', 'network', 'basic5', 'basic3', 'un2', 'boxes',
                'basic1', 'basic4']
    for i in range(datasets.__len__()):
        keys = [datasets[i]]
        data_path = "D:\\02_code\\01_Python\\Internal-index\\HB-Acceleration\\datasets\\data\\"
        hb_list_temp, data = hbc(keys, data_path)
        centers = np.array(get_all_centers(hb_list_temp))
        on, onCVI, ch_values = testSil(centers)
        print(f"粒球优化SIL指数为{onCVI:.2f},对应的簇数为{on}")
        print("=========粒球和原始分界线============")
        on, onCVI, ch_values = testSil(data)
        print(f"原始SIL指数为{onCVI:.2f},对应的簇数为{on}")
        print("--------------------------------------------")


if __name__ == '__main__':
    main()
