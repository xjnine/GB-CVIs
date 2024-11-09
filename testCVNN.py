from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import pairwise_distances
from HyperBallClustering_acceleration_v4 import *


def get_all_centers(hb):
    centers = []
    for ball in hb:
        if len(ball) > 2:
            center = ball.mean(0)
            centers.append(center)
    return centers


def testCVNN(D):
    k = 10
    A = pairwise_distances(D)
    NN = [None] * D.shape[0]
    kk = int(np.ceil(np.sqrt(D.shape[0])))

    C = []
    for ii in range(D.shape[0]):
        sorted_indices = np.argsort(A[:, ii])
        sorted_distances = A[sorted_indices, ii]
        C.append(np.vstack((sorted_distances, sorted_indices)).T)

    for ii in range(D.shape[0]):
        NN[ii] = C[ii][1:k + 1, :]

    seq = np.zeros(kk - 1)
    com = np.zeros(kk - 1)
    cvnn = np.zeros(kk - 1)

    Z = linkage(A, method='single')

    for ii in range(2, kk + 1):
        cl = fcluster(Z, ii, criterion='maxclust')
        seq1 = np.zeros(ii)
        com1 = np.zeros(ii)

        for mm in range(ii):
            a = np.where(cl == mm + 1)[0]
            ni = len(a)
            if ni > 1:
                com1[mm] = np.sum(pairwise_distances(D[a, :], D[a, :])) / (2 * (ni * (ni - 1) / 2))
            else:
                com1[mm] = 0
        com[ii - 2] = np.sum(com1)

        for jj in range(D.shape[0]):
            flag = cl[jj]
            temp = np.where(cl[NN[jj][:, 1].astype(int)] != flag)[0]
            if len(temp) != 0:
                seq1[flag - 1] += len(temp) / k

        avgseq1 = np.zeros(ii)
        for nn in range(ii):
            avgseq1[nn] = seq1[nn] / np.sum(cl == nn + 1) if np.sum(cl == nn + 1) > 0 else 0
        seq[ii - 2] = np.max(avgseq1)

    seq = seq / np.max(seq)
    com = com / np.max(com)

    cvnn = seq + com
    on = np.argmin(cvnn) + 2
    oncvi = np.min(cvnn)
    CL = fcluster(Z, on, criterion='maxclust')

    return on, oncvi, cvnn, CL


def main():
    datasets = ['n2', 'n3', 'face', 'D13', 'spirals', 'network', 'basic5', 'basic3', 'un2', 'boxes',
                'basic1', 'basic4']
    for i in range(datasets.__len__()):
        keys = [datasets[i]]
        data_path = "D:\\02_code\\01_Python\\Internal-index\\HB-Acceleration\\datasets\\data\\"
        hb_list_temp, data = hbc(keys, data_path)
        centers = np.array(get_all_centers(hb_list_temp))
        on, oncvi, cvnn, CL = testCVNN(centers)
        print(f"粒球优化CVNN指数为{oncvi:.2f},对应的簇数为{on}")
        print("=========粒球和原始分界线============")
        on, oncvi, cvnn, CL = testCVNN(data)
        print(f"原始CVNN指数为{oncvi:.2f},对应的簇数为{on}")
        print("--------------------------------------------")


if __name__ == '__main__':
    main()
