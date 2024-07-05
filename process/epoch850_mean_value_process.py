import numpy as np
import matplotlib.pyplot as plt

# 读取txt文件中的数据
data = {}
with open('data_2.txt', 'r') as file:
    for line in file:
        key, values = line.strip().split(':')
        values = list(map(float, values.split(',')))
        data[key] = values

# 提取diversity和threshold信息
diversities = sorted(set(label.split('_')[3] for label in data.keys()))
thresholds = sorted(set(label.split('_')[5] for label in data.keys()))

# 准备绘制图表的数据
plot_data = {thre: [] for thre in thresholds}
for thre in thresholds:
    for div in diversities:
        plot_data[thre].append(data[f'epoch_850_div_{div}_thre_{thre}'])

# 定义颜色
colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'lightpink', 'lightyellow', 'lightgrey', 'lightblue', 'lightcyan']
color_map = {thre: colors[i % len(colors)] for i, thre in enumerate(thresholds)}

# 自定义异常值的样式
flierprops = dict(marker='o', color='red', alpha=0.5)  # 你可以根据需要修改样式

# 绘制箱线图
plt.figure(figsize=(12, 8))
for i, thre in enumerate(thresholds):
    positions = np.arange(len(diversities)) + i * 0.2
    plt.boxplot(plot_data[thre], positions=positions, widths=0.15, patch_artist=True,
                boxprops=dict(facecolor=color_map[thre], color=color_map[thre]),
                medianprops=dict(color='black'),
                flierprops=flierprops,  # 使用自定义的异常值样式
                showfliers=True)  # 设置为False以隐藏异常值

# 设置x轴标签和图例
plt.xticks(np.arange(len(diversities)) + 0.2 * (len(thresholds) - 1) / 2, diversities)
plt.xlabel('Lambda Diversity')
plt.ylabel('Classification Accuracy of Face Recognition Model')
plt.title('Effect of Threshold and Lambda Diversity on Classification Accuracy of MaskNets Gengerated in One Batch')

# 创建图例
handles = [plt.Line2D([0], [0], color=color_map[thre], marker='s', markersize=10, label=f'threshold={thre}', linestyle='') for thre in thresholds]
plt.legend(handles=handles, title='Threshold')

plt.grid(True)
plt.tight_layout()
plt.show()
