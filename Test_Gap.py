import math
from scipy.spatial import distance
from sklearn.cluster import KMeans
import time
from Granular_Ball_Generation import *


def W_k(data, clusters):
    kmeans = KMeans(n_clusters=clusters, n_init=10, random_state=0).fit(data)
    w_k = 0
    for k in range(0, clusters):
        selected_data = data[kmeans.labels_ == k]
        dist_matrix = distance.cdist(selected_data, selected_data, metric='euclidean')
        sq_dist_matrix = dist_matrix ** 2
        sum_sq_distances = np.sum(sq_dist_matrix)
        n_r = sum(kmeans.labels_ == k)
        w_k += sum_sq_distances / (n_r << 1)
    return w_k


def Gapk_and_sk(data, clusters, B):
    sum_of_log_w_kb = 0
    w_kb_array = np.zeros(B)
    for b in range(B):
        dataB = np.random.uniform(np.min(data, axis=0), np.max(data, axis=0), data.shape)
        w_kb = W_k(dataB, clusters)
        w_kb_array[b] = w_kb
        sum_of_log_w_kb += math.log(w_kb, 2)
    sum_of_log_w_kb_divides_B = sum_of_log_w_kb / B
    gap_k = sum_of_log_w_kb_divides_B - math.log(W_k(data, clusters), 2)
    sd_k = 0
    for b in range(B):
        sd_k += (math.log(w_kb_array[b], 2) - sum_of_log_w_kb_divides_B) ** 2
    sd_k = (sd_k / B) ** 0.5
    s_k = sd_k * math.sqrt(1 + 1 / B)
    return gap_k, s_k


def get_all_centers(hb):
    centers = []
    for ball in hb:
        if len(ball) > 2:
            center = ball.mean(0)
            centers.append(center)
    return centers


def main():
    datasets_list = ['basic4']
    for i in range(datasets_list.__len__()):
        print("dataset：", datasets_list[i])
        keys = [datasets_list[i]]
        start_time_a = time.time()
        # Path to the dataset
        data_path = "./datasets/"
        hb_list, data = hbc(keys, data_path)
        centers = np.array(get_all_centers(hb_list))
        print('The number of granular balls：', len(centers))
        print('The number of original data：', len(data))
        nums_of_gbs = math.ceil(len(centers) ** 0.5)
        nums_of_origin = math.ceil(len(data) ** 0.5)
        # Using k-means clustering
        best_k = 0
        gap_k, sk = Gapk_and_sk(centers, 2, 10)
        for k in range(2, nums_of_gbs):
            gap_k_plus_1, sk_plus_1 = Gapk_and_sk(centers, k + 1, 10)
            if gap_k > gap_k_plus_1 + sk_plus_1:
                best_k = k
                break
            gap_k, sk = gap_k_plus_1, sk_plus_1
        end_time_a = time.time()
        t = end_time_a - start_time_a
        print(
            f'Optimized Gap algorithm with granular-ball, the best number of clusters obtained is:{best_k}, runTime is:{t:.2f}s')
        print("================================================")
        start_time_b = time.time()
        best_k = 0
        gap_k, sk = Gapk_and_sk(data, 2, 10)
        for k in range(2, nums_of_origin):
            gap_k_plus_1, sk_plus_1 = Gapk_and_sk(data, k + 1, 10)
            if gap_k > gap_k_plus_1 + sk_plus_1:
                best_k = k
                break
            gap_k, sk = gap_k_plus_1, sk_plus_1
        end_time_b = time.time()
        t = end_time_b - start_time_b
        print(
            f'The optimal number of clusters obtained from the original Gap algorithm is:{best_k}, runTime is:{t:.2f}s')
        print("--------------------------------------------")


if __name__ == '__main__':
    main()
