import numpy as np

with open("../output/dropout_new.txt", "r") as file:
    matrices = []
    current_matrix = []
    for line in file:
        if line.strip() == "":
            matrices.append(current_matrix)
            current_matrix = []
        else:
            values = line.strip().split()
            current_matrix.append([float(val) for val in values if val != "0"])

res_mean = []
res_variance = []
for i, matrix in enumerate(matrices):
    matrix = np.array(matrix)
    non_zero_values = matrix[matrix != 0]
    mean = np.mean(non_zero_values)
    res_mean.append(mean)
    variance = np.var(non_zero_values)
    res_variance.append(variance)
    print(f"Matrix {i+1}:")
    print(f"Mean: {mean}")
    print(f"Variance: {variance}")
    print()

average_mean = np.mean(res_mean)
average_variance = np.mean(res_variance)
print(f"Mean: {average_mean}")
print(f"Variance: {average_variance}")
