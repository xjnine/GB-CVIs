import numpy as np
import scipy.sparse as sp


def dijkstra(weight, start_node):
    # 将权重矩阵和起始节点调整为从0开始的索引
    weight = weight[1:, 1:]
    start_node -= 1

    n_nodes = weight.shape[0]

    # 初始化距离矩阵D和前驱矩阵Z
    D = np.full(n_nodes, np.inf)
    D[start_node] = 0
    Z = np.full(n_nodes, -1)

    # 初始化已访问节点列表
    visited_nodes = set()

    # 初始化掩码
    mask = np.ones(n_nodes, dtype=bool)

    while len(visited_nodes) < n_nodes:
        # 更新掩码
        mask[list(visited_nodes)] = False
        # 获取未访问节点的距离
        unvisited_distances = D[mask]
        # 在未访问的节点中找到距离起始节点最近的节点
        min_distance_node = np.where(mask)[0][np.argmin(unvisited_distances)]

        # 标记最近节点为已访问
        visited_nodes.add(min_distance_node)

        # 更新从起始节点到其他所有节点的最小距离和前驱节点
        for neighbor in weight.getrow(min_distance_node).nonzero()[1]:
            if neighbor not in visited_nodes:
                new_distance = D[min_distance_node] + weight[min_distance_node, neighbor]
                if new_distance < D[neighbor]:
                    D[neighbor] = new_distance
                    Z[neighbor] = min_distance_node

    # 将前驱矩阵Z中的值调整为从1开始的索引
    Z = np.array([z+1 if z!=-1 else -1 for z in Z])

    # 将结果的长度增加1，使其下标从1开始
    D = np.insert(D, 0, np.inf)
    Z = np.insert(Z, 0, -1)

    return D, Z

