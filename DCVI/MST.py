import matplotlib.pyplot as plt


def MST(D, result):
    plt.plot(D[:, 0], D[:, 1], 'r.', markersize=10)

    for i in range(result.shape[0]):
        x = [D[result[i, 0], 0], D[result[i, 1], 0]]
        y = [D[result[i, 0], 1], D[result[i, 1], 1]]
        plt.plot(x, y, 'bo-')

    plt.show()
