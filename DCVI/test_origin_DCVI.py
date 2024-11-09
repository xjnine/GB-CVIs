from origin_DCVI import *
import time
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


def DCVIClustering(D):
    result, D1, LPS, CP, CL, CPV = DCVI(D)
    on, onCVI, cl = TestCVI(result)
    CL[LPS] = cl

    for ii in range(D.shape[0]):
        if CL[ii] == 0:
            CL[ii] = CL[int(CP[ii])]
    return CL, on, onCVI, result, CP, CPV, LPS, cl


def main():
    datasets = ['n2', 'n3', 'face', 'D13', 'spirals', 'network', 'basic5', 'basic3', 'un2', 'boxes',
                'basic1', 'basic4']
    for i in range(datasets.__len__()):
        keys = [datasets[i]]
        data_path = "D:\\02_code\\01_Python\\Internal-index\\HB-Acceleration\\datasets\\data\\"
        st = time.time()
        df = pd.read_csv(data_path + keys[0] + ".csv", header=None)
        data = df.values
        if data.shape[1] >= 3:
            print("降维中....")
            data = StandardScaler().fit_transform(data)
            # data = MinMaxScaler().fit(data).transform(data)
            data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
            pca = PCA(n_components=2)
            data = pca.fit(data).transform(data)
            print("降维结束")
        else:
            data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
        CL, on, onCVI, result, CP, CPV, LPS, cl = DCVIClustering(data)
        ed = time.time()
        t = ed - st
        print(f"dataset is: {datasets[i]}, 原始DCVI经历了：{t:.2f}s , on is:{on}")
        print("-----------------------------------------------------")


if __name__ == '__main__':
    main()
