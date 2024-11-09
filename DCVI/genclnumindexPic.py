import matplotlib.pyplot as plt

x = list(range(2, 16))
a = [0.8553,
     0.3369,
     0.2751,
     0.3106,
     0.1168,
     0.0822,
     0.1132,
     0.1908,
     0.1979,
     0.2456,
     0.2284,
     0.2511,
     0.2381,
     0.2268]

plt.plot(x, a, '*-')
plt.axis([2, 15, 0.05, 0.9])
plt.xlabel('number of clusters')
plt.ylabel('DCVI index')
plt.gca().set_prop_cycle(None)  # 恢复默认颜色循环
plt.plot(7, 0.0822, 'r.', markersize=35)
plt.show()