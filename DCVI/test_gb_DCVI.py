from origin_DCVI import *
from HyperBallClustering_acceleration_v4 import *
import time


def get_all_centers(hb):
    centers = []
    for ball in hb:
        if len(ball) > 2:
            center = ball.mean(0)
            centers.append(center)
    return centers


def get_all_radius(hb):
    radius = []
    for ball in hb:
        if len(ball) > 2:
            radius.append(get_radius(ball))
    return radius


def DCVIClustering(D):
    result, D1, LPS, CP, CL, CPV = DCVI(D)
    on, onCVI, cl = TestCVI(result)
    CL[LPS] = cl

    for ii in range(D.shape[0]):
        if CL[ii] == 0:
            CL[ii] = CL[int(CP[ii])]
    return CL, on, onCVI, result, CP, CPV, LPS, cl


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def main():
    datasets = ['n2', 'n3', 'face', 'D13', 'spirals', 'network', 'basic5', 'basic3', 'un2', 'boxes',
                'basic1', 'basic4']

    for i in range(datasets.__len__()):
        keys = [datasets[i]]
        data_path = "D:\\02_code\\01_Python\\Internal-index\\HB-Acceleration\\datasets\\data\\"
        st = time.time()
        hb_list_temp, data = hbc(keys, data_path)
        centers = get_all_centers(hb_list_temp)
        centers = np.array(centers)

        CL, on, onCVI, result, CP, CPV, LPS, cl = DCVIClustering(centers)
        et = time.time()
        t = et - st
        print(f"dataset is: {datasets[i]}, 粒球优化DCVI经历了：{t:.2f}s ,, on is:{on},")
        print("-----------------------------------------------------")


if __name__ == '__main__':
    main()
