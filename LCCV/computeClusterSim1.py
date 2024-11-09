import numpy as np


def computeClusterSim1(cdata, conn, cluster_number, nc):
    # 计算簇间相似度，利用相对互连度和接近度
    sim = np.zeros((cluster_number + 1, cluster_number + 1))
    for i in range(1, 1 + cluster_number):
        for j in range(i + 1, cluster_number + 1):
            ncutedge = 0
            sumcutedge = 0
            for k in range(1, nc[i] + 1):
                for m in range(1, nc[j] + 1):
                    if conn[cdata[i][k], cdata[j][m]] != 0:
                        ncutedge += 1
                        sumcutedge += conn[cdata[i][k], cdata[j][m]]

            if ncutedge != 0:
                ave = sumcutedge / ncutedge
                sim[i][j] = ave ** 3 * ncutedge
                sim[j][i] = sim[i][j]
    return sim
