from mat2csv import read_file
from NTHC_Clustering import cluster
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

if __name__ == "__main__":
    # Data sets to be clustered (Rows of X correspond to points, columns correspond to variables)
    # input_file_name = './datasets/D5.mat'

    # K : The number of clusters
    K = 3
    # csv文件可以直接读数据，输入到cluster函数
    df = pd.read_csv("./datasets/D5.CSV", header=None)
    dataset = df.values

    # cluster
    # 读mat文件内容
    # dataset = read_file(input_file_name)
    # draw result
    clusters, label = cluster(dataset, K, ka=10, la=0.2, si=0.65)
    plt.scatter(clusters[:, 0], clusters[:, 1], c=label)
    # plt.savefig('D:\\02_code\\05_result\\LCCV\\gb_lccv_7.10\\' + keys[0] + '.png', dpi=1000, bbox_inches='tight')
    plt.show()
