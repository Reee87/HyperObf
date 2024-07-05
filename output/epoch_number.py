import matplotlib.pyplot as plt

# 数据
epoch_numbers = [500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150]
accuracy1 = [0.62, 0.65, 0.66, 0.64, 0.48, 0.45, 0.15, 0.16, 0.05, 0.09, 0.07, 0.09, 0.09, 0.09]
accuracy2 = [0.53, 0.49, 0.40, 0.46, 0.42, 0.39, 0.37, 0.35, 0.45, 0.49, 0.56, 0.59, 0.58, 0.61]

plt.figure(figsize=(10, 6))

# 画图
plt.plot(epoch_numbers, accuracy1, label='Accuracy in the absence of inversion attack')
plt.plot(epoch_numbers, accuracy2, label='Accuracy in the presence of inversion attack')

# 添加图例
plt.legend()

# 添加标签和标题
plt.xlabel('Training Epoch Number')
plt.ylabel('Classification Accuracy')
plt.title('Classification Accuracy vs. Training Epoch Number')

# 显示图形
plt.grid(True)
plt.show()
