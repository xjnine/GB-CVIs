from sklearn.metrics import calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, fcluster
from HyperBallClustering_acceleration_v4 import *


def get_all_centers(hb):
    centers = []
    for ball in hb:
        if len(ball) > 2:
            center = ball.mean(0)
            centers.append(center)
    return centers


def testCH(D):
    max_k = int(np.ceil(np.sqrt(D.shape[0])))
    k_list = range(2, max_k)

    optimal_k = None
    best_ch = -float('inf')
    ch_values = []

    for k in k_list:
        Z = linkage(D, method='ward')
        labels = fcluster(Z, k, criterion='maxclust')

        ch = calinski_harabasz_score(D, labels)
        ch_values.append(ch)

        if ch > best_ch:
            best_ch = ch
            optimal_k = k

    on = optimal_k
    onCVI = ch_values[optimal_k - 2]
    return on, onCVI, ch_values


def main():
    datasets = ['n2', 'n3', 'face', 'D13', 'spirals', 'network', 'basic5', 'basic3', 'un2', 'boxes',
                'basic1', 'basic4']
    for i in range(datasets.__len__()):
        print("此时数据集为：", i)
        keys = [datasets[i]]
        data_path = "D:\\02_code\\01_Python\\Internal-index\\HB-Acceleration\\datasets\\data\\"
        hb_list_temp, data = hbc(keys, data_path)
        centers = np.array(get_all_centers(hb_list_temp))
        on, onCVI, ch_values = testCH(centers)
        print(f"粒球优化CH指数为{onCVI:.2f},对应的簇数为{on},分数{ch_values}")
        print("=========粒球和原始分界线============")
        on, onCVI, ch_values = testCH(data)
        print(f"原始CH指数为{onCVI:.2f},对应的簇数为{on},分数{ch_values}")
        print("--------------------------------------------")


if __name__ == '__main__':
    main()
