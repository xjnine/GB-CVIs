from sklearn.metrics import davies_bouldin_score
from scipy.cluster.hierarchy import linkage, fcluster
from HyperBallClustering_acceleration_v4 import *


def get_all_centers(hb):
    centers = []
    for ball in hb:
        if len(ball) > 2:
            center = ball.mean(0)
            centers.append(center)
    return centers


def testDBI(D):
    max_k = int(np.ceil(np.sqrt(D.shape[0])))
    k_list = range(2, max_k)

    optimal_k = None
    best_dbi = float('inf')
    cvi_values = []

    for k in k_list:
        Z = linkage(D, method='ward')
        labels = fcluster(Z, k, criterion='maxclust')

        dbi = davies_bouldin_score(D, labels)
        cvi_values.append(dbi)
        print(k, dbi)
        if dbi < best_dbi:
            best_dbi = dbi
            optimal_k = k

    on = optimal_k
    onCVI = cvi_values[optimal_k - 2]
    return on, onCVI, cvi_values


def main():
    datasets = ['n2', 'n3', 'face', 'D13', 'spirals', 'network', 'basic5', 'basic3', 'un2', 'boxes',
                'basic1', 'basic4']
    for i in range(datasets.__len__()):
        keys = [datasets[i]]
        data_path = "D:\\02_code\\01_Python\\Internal-index\\HB-Acceleration\\datasets\\data\\"
        hb_list_temp, data = hbc(keys, data_path)
        centers = np.array(get_all_centers(hb_list_temp))
        on, onCVI, cvi_values = testDBI(centers)
        print(f"粒球优化DBI指数为{onCVI:.2f},对应的簇数为{on}")
        print("=========粒球和原始分界线============")
        on, onCVI, cvi_values = testDBI(data)
        print(f"原始DBI指数为{onCVI:.2f},对应的簇数为{on}")
        print("--------------------------------------------")


if __name__ == '__main__':
    main()
