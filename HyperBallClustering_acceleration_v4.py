# -*- coding: utf-8 -*-

from scipy.spatial.distance import pdist, squareform
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

"""
1、hbc
2、plot_dot
4、draw_ball
"""


def plot_dot(data, x):
    """

    :param data:
    :return:
    """
    if (x == 1):
        plt.scatter(data[:, 0], data[:, 1], s=7, c="#1f77b4", linewidths=5, marker='o', label='data point')
    else:
        plt.scatter(data[:, 0], data[:, 1], s=7, c="#8c564b", label='data point')


def draw_ball(hb_list):
    """
    :param hb_list:
    :return
    """
    is_isolated = False
    for data in hb_list:
        if len(data) > 1:
            center = data.mean(0)
            radius = np.max((((data - center) ** 2).sum(axis=1) ** 0.5))
            theta = np.arange(0, 2 * np.pi, 0.01)
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            plt.plot(x, y, ls='-', color='black', lw=0.7)
        else:
            plt.plot(data[0][0], data[0][1], marker='*', color='#f52324', markersize=3)
            is_isolated = True
    plt.plot([], [], ls='-', color='black', lw=1.2, label='hyper-ball boundary')
    if is_isolated:
        plt.scatter([], [], marker='*', color='#f52324', label='isolated point')
    plt.show()


def draw_line(point1, point2):
    plt.xlim(0.3, 0.6)
    plt.ylim(0.5, 0.8)
    plt.plot([point1[0], point2[0]], [point1[1], point2[1]], 'ro-', label='line')


def get_dm(hb):
    num = len(hb)
    if (num == 0):
        return 0
    center = hb.mean(0)
    diff_mat = center - hb
    sq_diff_mat = diff_mat ** 2

    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    sum_radius = sum(distances)
    mean_radius = sum_radius / num
    if num > 2:
        return mean_radius
    else:
        return 1


def division(hb_list, hb_list_not):
    gb_list_new = []
    for hb in hb_list:
        if len(hb) > 0:
            ball_1, ball_2 = spilt_ball(hb)
            dm_parent = get_dm(hb)
            dm_child_1 = get_dm(ball_1)
            dm_child_2 = get_dm(ball_2)
            w = len(ball_1) + len(ball_2)
            w1 = len(ball_1) / w
            w2 = len(ball_2) / w
            w_child = w1 * dm_child_1 + w2 * dm_child_2
            t2 = w_child < dm_parent
            if t2:
                gb_list_new.extend([ball_1, ball_2])
            else:
                hb_list_not.append(hb)
        else:
            hb_list_not.append(hb)
    return gb_list_new, hb_list_not


def spilt_ball1(data):
    ball1 = []
    ball2 = []
    A = pdist(data)
    d_mat = squareform(A)
    r, c = np.where(d_mat == np.max(d_mat))

    r1 = r[1]
    c1 = c[1]
    temp1 = d_mat[:, r1] < d_mat[:, c1]
    temp2 = d_mat[:, r1] >= d_mat[:, c1]
    ball1 = data[temp1, :]
    ball2 = data[temp2, :]

    ball1 = np.array(ball1)
    ball2 = np.array(ball2)
    return [ball1, ball2]


def spilt_ball(data):
    center = data.mean(0)
    n, d = np.shape(data)
    dist_1_mat = np.sqrt(np.sum(np.asarray(center - data) ** 2, axis=1).astype('float'))  # 离中心最远点之间距离矩阵
    index_1_mat = np.where(dist_1_mat == np.max(dist_1_mat))
    if len(data[index_1_mat, :][0]) >= 2:
        p1 = np.reshape(data[index_1_mat, :][0][0], [d, ])
    else:
        p1 = np.reshape(data[index_1_mat, :], [d, ])
    dist_2_mat = np.sqrt(np.sum(np.asarray(p1 - data) ** 2, axis=1).astype('float'))
    index_2_mat = np.where(dist_2_mat == np.max(dist_2_mat))
    if len(data[index_2_mat, :][0]) >= 2:
        p2 = np.reshape(data[index_2_mat, :][0][0], [d, ])
    else:
        p2 = np.reshape(data[index_2_mat, :], [d, ])

    c_p1 = (center + p1) / 2
    c_p2 = (center + p2) / 2

    dist_p1 = np.linalg.norm(data - c_p1, axis=1)
    dist_p2 = np.linalg.norm(data - c_p2, axis=1)

    ball1 = data[dist_p1 <= dist_p2]
    ball2 = data[dist_p1 > dist_p2]

    return [ball1, ball2]


def get_radius(hb):
    num = len(hb)
    center = hb.mean(0)
    diff_mat = center - hb
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    radius = max(distances)
    return radius


def normalized_ball(hb_list, hb_list_not, radius_detect, radius, whileflag=0):
    hb_list_temp = []
    if whileflag != 1:
        for hb in hb_list:
            if len(hb) < 2:
                hb_list_not.append(hb)
            else:
                if get_radius(hb) <= 2 * radius_detect:
                    hb_list_not.append(hb)
                else:
                    ball_1, ball_2 = spilt_ball(hb)
                    hb_list_temp.extend([ball_1, ball_2])
    if whileflag == 1:
        for i, hb in enumerate(hb_list):
            if len(hb) < 2:
                hb_list_not.append(hb)
            else:
                if radius[i] <= 2 * radius_detect:
                    hb_list_not.append(hb)
                else:
                    ball_1, ball_2 = spilt_ball(hb)
                    hb_list_temp.extend([ball_1, ball_2])
    return hb_list_temp, hb_list_not


def hbc(keys, data_path):
    looptime = 1
    for d in range(len(keys)):
        df = pd.read_csv(data_path + keys[d] + ".csv", header=None)
        data = df.values
        start_time = datetime.now()
        if data.shape[1] >= 3:
            data = StandardScaler().fit(data).transform(data)
            data = MinMaxScaler().fit(data).transform(data)
            pca = PCA(n_components=2)
            data = pca.fit(data).transform(data)
        else:
            data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
        data = np.unique(data, axis=0)
        hb_list_temp = [data[1:, :]]
        hb_list_not_temp = []
        while 1:
            ball_number_old = len(hb_list_temp) + len(hb_list_not_temp)
            hb_list_temp, hb_list_not_temp = division(hb_list_temp, hb_list_not_temp)
            ball_number_new = len(hb_list_temp) + len(hb_list_not_temp)
            if ball_number_new == ball_number_old:
                hb_list_temp = hb_list_not_temp
                break
        radius = []
        hb_list_temp2 = []
        for hb in hb_list_temp:
            if len(hb) >= 2:
                hb_list_temp2.append(hb)
                radius.append(get_radius(hb))
        hb_list_temp = hb_list_temp2
        radius_median = np.median(radius)
        radius_mean = np.mean(radius)

        radius_detect = max(radius_median, radius_mean)
        hb_list_not_temp = []
        while 1:
            ball_number_old = len(hb_list_temp) + len(hb_list_not_temp)
            hb_list_temp, hb_list_not_temp = normalized_ball(hb_list_temp, hb_list_not_temp, radius_detect, radius,
                                                             whileflag=looptime)
            looptime = looptime + 1
            ball_number_new = len(hb_list_temp) + len(hb_list_not_temp)
            if ball_number_new == ball_number_old:
                hb_list_temp = hb_list_not_temp
                break
        end_time = datetime.now()
        print("dataset：", keys[d], ",time ：", end_time - start_time)
    return hb_list_temp, data
