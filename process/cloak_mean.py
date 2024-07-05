import re
import matplotlib.pyplot as plt

# 读取文本数据
file_path = "../output/sensitivity_variance.txt"  # 替换为您的文件路径
with open(file_path, "r") as file:
    text = file.read()

# 正则表达式模式匹配参数组合和方差值
pattern = r"The variance value of the parameters epoch_(\d+)_div_(\d+)_thre_0.01 is: (\S+)"
matches = re.findall(pattern, text)

# 存储参数组合和方差值的字典
variance_data = {"div_1": [], "div_10": [], "div_100": []}

for match in matches:
    epoch = int(match[0])
    div = int(match[1])
    # 去掉末尾的逗号并拆分字符串
    variances_str = match[2].rstrip(',').split(", ")
    variances = [float(val) for val in variances_str]  # 将方差值从字符串转换为列表
    average_variance = sum(variances) / len(variances)  # 计算方差值的平均值
    if div == 1:
        variance_data["div_1"].append((epoch, average_variance))
    elif div == 10:
        variance_data["div_10"].append((epoch, average_variance))
    elif div == 100:
        variance_data["div_100"].append((epoch, average_variance))

# 打印结果
for div, data in variance_data.items():
    print(f"div_{div}:")
    for epoch, variance in data:
        print(f"Epoch {epoch}: Average Variance = {variance}")

# 对参数按照 epoch 进行排序
for data_list in variance_data.values():
    data_list.sort(key=lambda x: x[0])

# 绘制曲线图
plt.figure(figsize=(12, 6))

for div, data in variance_data.items():
    if data:  # 检查列表是否为空
        epochs, variances = zip(*data)
        plt.plot(epochs, variances, label=f"diversity lambda = {div.split('_')[1]}")

plt.xlabel('Epoch number')
plt.ylabel('Variance of model classification accuracy trained on perturbed images')
plt.title('Variance of model classification accuracy trained on perturbed images with diversity threshold = 0.01')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../output/variances_001.png")
plt.show()


