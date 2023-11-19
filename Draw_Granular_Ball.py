# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 22:13:04 2023
@author: xjnine
"""
from Granular_Ball_Generation import *


def get_all_centers(hb):
    centers = []
    for ball in hb:
        if len(ball) > 2:
            center = ball.mean(0)
            centers.append(center)
    return centers


def main():
    datasets_list = ['face', 'basic4', 'boxes']
    data_path = "./datasets/"
    for i in range(datasets_list.__len__()):
        keys = [datasets_list[i]]
        hb_result_list, data_result_list = hbc(keys, data_path)
        plot_dot(data_result_list)
        draw_ball(hb_result_list)
        plt.show()


if __name__ == '__main__':
    main()
