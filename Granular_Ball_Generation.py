# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 21:46:00 2022

@author: xjnine
"""

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

"""
1、hbc
2、plot_dot
4、draw_ball
"""


def plot_dot(data):
    plt.figure(figsize=(6, 6))
    plt.scatter(data[:, 0], data[:, 1], s=7, c="#314300", linewidths=5, alpha=0.6, marker='o', label='data point')
    plt.legend(loc=1)


def draw_ball(hb_list):
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
            plt.plot(data[0][0], data[0][1], marker='*', color='#0000EF', markersize=3)
            is_isolated = True
    plt.plot([], [], ls='-', color='black', lw=1.2, label='ball boundary')
    plt.legend(loc=1)
    if is_isolated:
        plt.scatter([], [], marker='*', color='#0000EF', label='isolated ball')
        plt.legend(loc=1)


def draw_line(point1, point2):
    plt.xlim(0.3, 0.6)
    plt.ylim(0.5, 0.8)
    plt.plot([point1[0], point2[0]], [point1[1], point2[1]], 'ro-', label='line')


# Calculating the quality of the sub-ball, where 'hb' represents all the points in the granular ball.
def get_dm(hb):
    num = len(hb)
    center = hb.mean(0)
    diff_mat = center - hb
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    sum_radius = sum(distances)  # 下面的语句没有必要使用for循环
    mean_radius = sum_radius / num
    if num > 2:
        return mean_radius
    else:
        return 1


# Deciding whether to split the ball based on the DM value
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
            w_child = w1 * dm_child_1 + w2 * dm_child_2  # The DM values of the two split child balls.
            if w_child < dm_parent:
                gb_list_new.extend([ball_1, ball_2])
            else:
                hb_list_not.append(hb)
        else:
            hb_list_not.append(hb)
    return gb_list_new, hb_list_not


# Splitting the ball
def spilt_ball(data):
    ball1 = []
    ball2 = []
    center = data.mean(0)
    n, d = np.shape(data)
    dist_1_mat = np.sqrt(np.sum(np.asarray(center - data) ** 2, axis=1).astype('float'))
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


# Calculating the radius of the ball
def get_radius(hb):
    num = len(hb)
    center = hb.mean(0)
    diff_mat = center - hb
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    radius = max(distances)
    return radius


# Normalizing balls
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
    hb_result_list = []
    data_result_list = []
    for d in range(len(keys)):
        df = pd.read_csv(data_path + keys[d] + ".csv", header=None)
        data = df.values
        data = np.unique(data.astype('float'), axis=0)
        data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
        start_time = datetime.datetime.now()
        hb_list_temp = [data]
        hb_list_not_temp = []
        # Division based on quality.
        while 1:
            ball_number_old = len(hb_list_temp) + len(hb_list_not_temp)
            hb_list_temp, hb_list_not_temp = division(hb_list_temp, hb_list_not_temp)
            ball_number_new = len(hb_list_temp) + len(hb_list_not_temp)
            if ball_number_new == ball_number_old:
                hb_list_temp = hb_list_not_temp
                break
        # Obtaining the radius of each ball
        radius = []
        for hb in hb_list_temp:
            if len(hb) >= 2:
                radius.append(get_radius(hb))
        radius_median = np.median(radius)
        radius_mean = np.mean(radius)
        radius_detect = max(radius_median, radius_mean)
        hb_list_not_temp = []
        # Normalization balls
        while 1:
            ball_number_old = len(hb_list_temp) + len(hb_list_not_temp)
            hb_list_temp, hb_list_not_temp = normalized_ball(hb_list_temp, hb_list_not_temp, radius_detect, radius,
                                                             whileflag=looptime)
            looptime = looptime + 1
            ball_number_new = len(hb_list_temp) + len(hb_list_not_temp)
            if ball_number_new == ball_number_old:
                hb_list_temp = hb_list_not_temp
                break
        end_time = datetime.datetime.now()
        print("dataset：", keys[d], ",The generation time for granular balls ：", end_time - start_time, 's')
        hb_result_list.append(hb_list_temp)
        data_result_list.append(data)
    return hb_list_temp, data
