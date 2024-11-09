# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 21:46:00 2022

@author: xjnine
"""

from scipy.spatial.distance import pdist, squareform
import sklearn
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# 利用PCA降维，降到二维
from sklearn.decomposition import PCA

"""
1、hbc
2、plot_dot
4、draw_ball
"""


def plot_dot(data):
    """

    :param data:
    :return:
    """
    # plt.figure(figsize=(10, 10))
    # 生成scatter 散点图 data[a,b] a：行 b：列
    plt.scatter(data[:, 0], data[:, 1], s=7, c="#314300", linewidths=5, alpha=0.6, marker='o', label='data point')
    # plt.legend()的作用：
    # 在plt.plot() 定义后plt.legend() 会显示该 label 的内容
    plt.legend(loc=1)


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
            plt.plot(data[0][0], data[0][1], marker='*', color='#0000EF', markersize=3)
            is_isolated = True
    plt.plot([], [], ls='-', color='black', lw=1.2, label='hyper-ball boundary')
    plt.legend(loc=1)
    if is_isolated:
        plt.scatter([], [], marker='*', color='#0000EF', label='isolated point')
        plt.legend(loc=1)
    plt.show()


def draw_line(point1, point2):
    plt.xlim(0.3, 0.6)
    plt.ylim(0.5, 0.8)
    # plt.figure(figsize=(10, 10))
    plt.plot([point1[0], point2[0]], [point1[1], point2[1]], 'ro-', label='line')
    # plt.legend(loc=1)
    # plt.show()


def get_dm(hb):  # 计算子球的质量,hb为粒球中所有的点
    num = len(hb)
    if num == 0:
        return 0
    # 对hb数组按列计算均值，返回一个一维数组，表示每一列的平均值。
    # 它等价于np.mean(hb, axis=0)，其中axis=0表示按列计算。
    center = hb.mean(0)
    diff_mat = center - hb
    sq_diff_mat = diff_mat ** 2
    # 这行代码是对平方差矩阵进行计算，对每个粒子，求其与其他粒子的距离的平方，
    # 最后得到一个一维数组，表示每个粒子与其他粒子的距离的平方。
    # 这里的 axis=1 参数表示按照每一行进行求和。
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    sum_radius = 0
    sum_radius = sum(distances)  # 下面的语句没有必要使用for循环
    # for i in distances:
    #     sum_radius = sum_radius + i

    mean_radius = sum_radius / num
    # print('%%%%%%%mean_radius:', mean_radius)
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
            w_child = w1 * dm_child_1 + w2 * dm_child_2  # 某一个子球的质量
            # print('dm_parent',dm_parent)
            # print('len_child_1',len(ball_1))
            # print('dm_child_1',dm_child_1)
            # print('dm_child_2',dm_child_2)
            # print('np.shape(dm_child_1),np.shape(dm_child_2)',dm_child_1,dm_child_2)
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
    # n, m = data.shape
    # x_mat = data.T
    # g_mat = np.dot(x_mat.T, x_mat)
    # h_mat = np.tile(np.diag(g_mat), (n, 1))
    # d_mat = np.sqrt(h_mat + h_mat.T - g_mat * 2)
    # 调用pdist计算距离矩阵
    # pdist 认为输入矩阵中，每一行是一个实例。其计算每两行的数据，然后存储到A中。

    A = pdist(data)
    # squ用来压缩矩阵
    d_mat = squareform(A)
    # 行row 列colunm
    # 返回的结果是一个元组(r,c)，其中r是最大值所在的行数，c是最大值所在的列数。
    r, c = np.where(d_mat == np.max(d_mat))

    r1 = r[1]
    c1 = c[1]
    # 这里是根据距离矩阵中选出的两个最远的点 r1, c1 进行进一步的分类，
    # 将数据点分为两部分。
    # d_mat[:, r1] 是 d_mat 中所有行（即:）的第 r1 个元素。
    # 可以理解为选取了 d_mat 中以 r1 为列号的一列数据。
    # temp1 中，每个元素表示该数据点到 r1 的距离是否小于该点到 c1 的距离，
    # 如果是，则该数据点被分到一个球中，该球以 r1 为球心，以该点到 r1 的距离为半径；
    # 否则该点被分到另一个球中，该球以 c1 为球心，以该点到 c1 的距离为半径。
    # temp2 则与 temp1 正好相反，
    # 是将数据点分到另外一个球中，以 c1 为球心，以该点到 c1 的距离为半径。
    # temp1 temp2 都是bool型
    temp1 = d_mat[:, r1] < d_mat[:, c1]
    temp2 = d_mat[:, r1] >= d_mat[:, c1]
    # ball1.extend([data[temp1, :]])
    # ball2.extend([data[temp2, :]])
    # temp1是一个bool型的一维数组，它表示哪些行属于球1，哪些行属于球2。
    # temp2则是temp1的反向，它表示哪些行不属于球1，哪些行不属于球2。
    # 然后，根据temp1和temp2分别从data中选出属于球1的行和属于球2的行，即ball1和ball2。
    ball1 = data[temp1, :]
    ball2 = data[temp2, :]

    # for j in range(0, len(data)): #使用上面四行代码替代该循环
    #     if d_mat[j, r1] < d_mat[j, c1]:
    #         ball1.extend([data[j, :]])
    #     else:
    #         ball2.extend([data[j, :]])
    ball1 = np.array(ball1)
    ball2 = np.array(ball2)
    return [ball1, ball2]


def spilt_ball(data):
    center = data.mean(0)
    n, d = np.shape(data)
    dist_1_mat = np.sqrt(np.sum(np.asarray(center - data) ** 2, axis=1).astype('float'))  # 离中心最远点之间距离矩阵
    index_1_mat = np.where(dist_1_mat == np.max(dist_1_mat))  # 离中心最远点下标矩阵
    if len(data[index_1_mat, :][0]) >= 2:  # 如果存在多个最远点下标
        p1 = np.reshape(data[index_1_mat, :][0][0], [d, ])  # 取第一个最远点
    else:
        p1 = np.reshape(data[index_1_mat, :], [d, ])
    dist_2_mat = np.sqrt(np.sum(np.asarray(p1 - data) ** 2, axis=1).astype('float'))  # 离p1最远点之间距离矩阵
    index_2_mat = np.where(dist_2_mat == np.max(dist_2_mat))  # 离p1最远点下标矩阵
    if len(data[index_2_mat, :][0]) >= 2:
        p2 = np.reshape(data[index_2_mat, :][0][0], [d, ])  # 取第一个最远点
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
                # print('小循环radiusradiusradiusradius', get_radius(hb))
                if get_radius(hb) <= 2 * radius_detect:
                    hb_list_not.append(hb)
                else:
                    ball_1, ball_2 = spilt_ball(hb)
                    hb_list_temp.extend([ball_1, ball_2])
    if whileflag == 1:
        # enumerate 函数可以同时获得元素的索引和值。
        # 循环中的 i 是当前元素的索引，hb 是当前元素的值。
        for i, hb in enumerate(hb_list):
            if len(hb) < 2:
                hb_list_not.append(hb)
            else:
                # print('小循环radiusradiusradiusradius', get_radius(hb))
                # print(np.shape(radius),i)
                if radius[i] <= 2 * radius_detect:
                    hb_list_not.append(hb)
                else:
                    ball_1, ball_2 = spilt_ball(hb)
                    hb_list_temp.extend([ball_1, ball_2])
    return hb_list_temp, hb_list_not


def hbc(keys, data_path):
    looptime = 1  # normalized_ball函数中，控制第一次的getradius调用两次
    # print(len(keys))
    for d in range(len(keys)):
        # 这句话是在Python中读取CSV文件的代码。
        # 具体来说，它使用pandas库中的read_csv函数从指定路径的CSV文件中读取数据，
        # 并将其存储为一个pandas DataFrame对象，其中data_path是文件路径，
        # keys[d]是文件名，".csv"表示文件类型是CSV格式。
        # header=None表示CSV文件中没有列名，因此数据将从第一行开始读取。
        # 读取后的DataFrame可以进一步进行数据处理和分析。
        df = pd.read_csv(data_path + keys[d] + ".csv", header=None)
        # 获取数据
        # DataFrame对象df转换为numpy的ndarray数组
        data = df.values
        print(f"数据集大小为{len(data)}")
        # np.unique() 函数 去除其中重复的元素 ，
        # 并按元素 由小到大 返回一个新的无元素重复的元组或者列表。
        # 其中，axis=0表示对每一行进行去重，保留唯一的行。

        # Min数据归一化 fit对数据进行预处理 fit_tans即包含了训练又包含了转换
        if data.shape[1] >= 3:
            print('降维中....')
            data = StandardScaler().fit(data).transform(data)
            data = MinMaxScaler().fit(data).transform(data)
            # -----------------------PCA降维-------------
            # 设置参数n_components=2维度为2
            pca = PCA(n_components=2)
            # 传入数据
            data = pca.fit(data).transform(data)
            print("降维完成")
            # tsne = TSNE(n_components=2)
            # # 传入数据
            # centers = tsne.fit_transform(centers)
        # Min数据归一化 fit对数据进行预处理 fit_tans即包含了训练又包含了转换
        else:
            data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
        # data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
        data = np.unique(data, axis=0)

        start_time = datetime.now()
        # 这里的data指的是经过预处理后的数据。
        # hb_list_temp列表中只包含了一个元素，即data，
        # 表示将所有数据点看作一个单独的超球。
        hb_list_temp = [data]
        hb_list_not_temp = []
        # 按照质量分化
        while 1:
            ball_number_old = len(hb_list_temp) + len(hb_list_not_temp)
            hb_list_temp, hb_list_not_temp = division(hb_list_temp, hb_list_not_temp)
            ball_number_new = len(hb_list_temp) + len(hb_list_not_temp)
            if ball_number_new == ball_number_old:
                hb_list_temp = hb_list_not_temp
                break

                # 全局归一化
        hb_list_temp2 = []
        radius = []
        for hb in hb_list_temp:
            if len(hb) >= 2:
                hb_list_temp2.append(hb)
                radius.append(get_radius(hb))
        # print('大循环radiusradiusradiusradius',radius)
        # 我们将半径数组传入np.median()函数中，将计算得到的中位数赋值给变量radius_median。
        print('hb', len(hb_list_temp))
        print('radius', len(radius))

        hb_list_temp = hb_list_temp2
        radius_median = np.median(radius)
        # np.mean(radius)的作用是计算这些圆的半径的平均值。
        radius_mean = np.mean(radius)

        radius_detect = max(radius_median, radius_mean)
        hb_list_not_temp = []
        while 1:
            ball_number_old = len(hb_list_temp) + len(hb_list_not_temp)
            hb_list_temp, hb_list_not_temp = normalized_ball(hb_list_temp, hb_list_not_temp, radius_detect, radius,
                                                             whileflag=looptime)
            # looptime控制第一次的getradius调用两次
            looptime = looptime + 1
            ball_number_new = len(hb_list_temp) + len(hb_list_not_temp)
            if ball_number_new == ball_number_old:
                hb_list_temp = hb_list_not_temp
                break
        end_time = datetime.now()
        t = end_time - start_time
        print("数据集：", keys[d], ",生成粒球时间 ：", t)
    return hb_list_temp, data
