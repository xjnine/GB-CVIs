from HyperBallClustering_acceleration_v4 import *
import time
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import pairwise_distances


def calculate_wk(D, labels):
    unique_labels = np.unique(labels)
    Wk = 0
    for label in unique_labels:
        cluster_points = D[labels == label]
        if len(cluster_points) > 1:
            distances = pairwise_distances(cluster_points)
            Wk += np.sum(distances) / (2 * len(cluster_points))
    return Wk


def calculate_gap_statistic(D, B=10, max_k=None):
    if max_k is None:
        max_k = int(np.ceil(np.sqrt(D.shape[0])))
    k_list = range(2, max_k)

    n_samples = D.shape[0]
    log_wks = np.zeros(len(k_list))
    log_wk_refs = np.zeros((len(k_list), B))

    for i, k in enumerate(k_list):
        Z = linkage(D, method='ward')
        labels = fcluster(Z, k, criterion='maxclust')
        Wk = calculate_wk(D, labels)
        log_wks[i] = np.log(Wk)

    for b in range(B):
        reference = np.random.uniform(np.min(D, axis=0), np.max(D, axis=0), size=D.shape)
        for i, k in enumerate(k_list):
            Z_ref = linkage(reference, method='ward')
            labels_ref = fcluster(Z_ref, k, criterion='maxclust')
            Wk_ref = calculate_wk(reference, labels_ref)
            log_wk_refs[i, b] = np.log(Wk_ref)

    log_wk_refs_mean = np.mean(log_wk_refs, axis=1)
    gap_values = log_wk_refs_mean - log_wks

    optimal_k = k_list[np.argmax(gap_values)]

    return optimal_k, gap_values


def testGap(D):
    max_k = int(np.ceil(np.sqrt(D.shape[0])))  # 根据数据大小确定最大K值
    on, gap_values = calculate_gap_statistic(D, max_k=max_k)

    # 返回最佳K值，最佳Gap值，以及所有K值的Gap统计结果
    onCVI = np.max(gap_values)  # 最佳K对应的Gap值
    eva = gap_values  # 返回所有Gap值

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
        print("此时数据集为：", i)
        keys = [datasets[i]]
        data_path = "D:\\02_code\\01_Python\\Internal-index\\HB-Acceleration\\datasets\\data\\"
        hb_list_temp, data = hbc(keys, data_path)
        centers = np.array(get_all_centers(hb_list_temp))
        on, onCVI, eva = testGap(centers)
        print(f"粒球优化Gap指数为{onCVI:.2f},对应的簇数为{on}")
        print("=========粒球和原始分界线============")
        on, onCVI, eva = testGap(data)
        print(f"原始Gap指数为{onCVI:.2f},对应的簇数为{on}")
        print("--------------------------------------------")


if __name__ == '__main__':
    main()
