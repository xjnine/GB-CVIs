from test_origin_LCCV import HC_LCCV
from HyperBallClustering_acceleration_v4 import *
import time


def get_all_centers(hb):
    centers = []
    for ball in hb:
        if len(ball) > 2:
            center = ball.mean(0)
            centers.append(center)
    return centers


if __name__ == '__main__':

    datasets = ['n2', 'n3', 'face', 'D13', 'spirals', 'network', 'basic5', 'basic3', 'un2', 'boxes',
                'basic1', 'basic4']
    for i in range(datasets.__len__()):
        keys = [datasets[i]]
        st = time.time()
        data_path = "D:\\02_code\\01_Python\\Internal-index\\HB-Acceleration\\datasets\\data\\"
        hb_list_temp, data = hbc(keys, data_path)
        centers = get_all_centers(hb_list_temp)
        centers = np.array(centers)
        rows, cols = centers.shape
        new_array = np.zeros((rows + 1, cols + 1), dtype=centers.dtype)
        new_array[1:, 1:] = centers
        centers = new_array

        bestcl = HC_LCCV(centers)
        bestcl = bestcl[1:, 1:]
        et = time.time()
        t = et - st
        unique_elements = np.unique(bestcl)
        on = len(unique_elements)
        print(f"dataset is: {datasets[i]}, 粒球优化LCCV经历了：{t:.2f}s , on is:{on}")
        print('-----------------------------------------------------------------------------')
