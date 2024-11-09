from scipy.cluster.hierarchy import linkage, fcluster
import networkx as nx
from HyperBallClustering_acceleration_v4 import *


def get_all_centers(hb):
    centers = []
    for ball in hb:
        if len(ball) > 2:
            center = ball.mean(0)
            centers.append(center)
    return centers


def testCSP(D):
    k = D.shape[0]
    kk = int(np.ceil(np.sqrt(k)))

    Distance = squareform(pdist(D))

    Z = linkage(Distance, method='single')

    avgCSP = np.zeros(kk - 1)
    cl = np.zeros((kk - 1, k))

    for ii in range(2, kk + 1):
        CL = fcluster(Z, ii, criterion='maxclust')
        cd = np.zeros(ii)
        sd = np.zeros(ii)
        csp = np.zeros(ii)

        for mm in range(ii):
            a = np.where(CL == mm + 1)[0]
            b = np.where(CL != mm + 1)[0]

            D1 = Distance[np.ix_(a, a)]
            D2 = Distance[np.ix_(a, b)]

            if len(a) > 1:
                G = nx.Graph(D1)
                T = nx.minimum_spanning_tree(G)
                weight = [d['weight'] for (u, v, d) in T.edges(data=True)]
                cd[mm] = sum(weight) / (len(a) - 1)
            else:
                cd[mm] = 0

            sd[mm] = np.min(D2) if len(D2) > 0 else 0

        for tt in range(ii):
            csp[tt] = (sd[tt] - cd[tt]) / (sd[tt] + cd[tt]) if (sd[tt] + cd[tt]) != 0 else 0

        avgCSP[ii - 2] = np.mean(csp)
        cl[ii - 2, :] = CL

    on = np.argmax(avgCSP) + 2
    oncvi = np.max(avgCSP)
    CL = fcluster(Z, on, criterion='maxclust')

    return on, oncvi, avgCSP, CL


def main():
    datasets = ['n2', 'n3', 'face', 'D13', 'spirals', 'network', 'basic5', 'basic3', 'un2', 'boxes',
                'basic1', 'basic4']
    for i in range(datasets.__len__()):
        keys = [datasets[i]]
        data_path = "D:\\02_code\\01_Python\\Internal-index\\HB-Acceleration\\datasets\\data\\"
        hb_list_temp, data = hbc(keys, data_path)
        centers = np.array(get_all_centers(hb_list_temp))
        print("--------------------------------------------")
        on, oncvi, avgCSP, CL = testCSP(centers)
        print(f"最大的CSP指数为{oncvi:.2f},对应的簇数为{on}")
        print("=========粒球和原始分界线============")
        on, oncvi, avgCSP, CL = testCSP(data)
        print(f"原始最大的CSP指数为{oncvi:.2f},对应的簇数为{on}")


if __name__ == '__main__':
    main()
