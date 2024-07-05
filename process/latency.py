# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle

# # latency 数据
# latency = [
#     0.08310317993164062,
#     0.09581208229064941,
#     0.10649609565734863,
#     0.11817193031311035,
#     0.13170242309570312,
#     0.1434013843536377,
#     0.15500354766845703,
#     0.16712212562561035,
#     0.17767119407653809,
#     0.1924285888671875
# ]

# # 对应的误差数据
# error = [
#     0.0031,
#     0.0025,
#     0.002,
#     0.0025,
#     0.003,
#     0.0035,
#     0.004,
#     0.0035,
#     0.0025,
#     0.0035
# ]

# # 对应的 x 轴数据
# steps = list(range(10, 110, 10))

# # 创建画布和子图
# fig, ax = plt.subplots()

# # 绘制误差线
# for x, y, e in zip(steps, latency, error):
#     ax.add_patch(Rectangle((x - 0.5, y - e), 1, 2 * e, edgecolor='black', facecolor='lightblue'))

# # 绘制折线
# plt.plot(steps, latency, marker='o', color='blue')

# # 添加标题和标签
# plt.title('Latency Over Number of Models with Error Bars')
# plt.xlabel('Number of Models')
# plt.ylabel('Latency/Second')

# # 显示图形
# plt.show()

import matplotlib.pyplot as plt

# 数据
data = {
    10: 0.08310317993164062,
    100: 0.1924285888671875,
    1000: 1.3280434608459473,
    10000: 11.930733442306519
}

# 提取 x 和 y 数据
x = list(data.keys())
y = list(data.values())

# 绘制图形
plt.semilogx(x, y, marker='o')

# 添加标题和标签
plt.title('Latency Over Number of Models')
plt.xlabel('Number of Models')
plt.ylabel('Latency/Second')

# 显示图形
plt.show()

