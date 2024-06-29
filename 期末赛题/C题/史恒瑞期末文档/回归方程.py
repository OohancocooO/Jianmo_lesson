import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 读取txt文件
file_path = "C:\\Users\\86184\\Desktop\\每日加成定价与销量.txt"
data = pd.read_csv(file_path, delimiter='\t')

# 将NaN值替换为0
data = data.fillna(0)

# 创建线性回归模型
def fit_linear_regression(x, y):
    model = LinearRegression()
    model.fit(x, y)
    return model

# 拟合每一组的线性回归
models = []
for i in range(6):
    y = data.iloc[:, 1 + i * 2].values.reshape(-1, 1)  # 将成本加成定价作为y轴
    x = data.iloc[:, 2 + i * 2].values.reshape(-1, 1)  # 将销量作为x轴
    
    model = fit_linear_regression(x, y)
    models.append(model)
    print(f"Group {i + 1} Linear Regression Equation: y = {model.coef_[0][0]:.2f}x + {model.intercept_[0]:.2f}")

# 可视化结果
plt.figure(figsize=(18, 12))
for i in range(6):
    y = data.iloc[:, 1 + i * 2].values.reshape(-1, 1)
    x = data.iloc[:, 2 + i * 2].values.reshape(-1, 1)
    
    plt.subplot(2, 3, i + 1)
    plt.scatter(x, y, color='blue')
    if i < len(models):
        plt.plot(x, models[i].predict(x), color='red')
    plt.title(f'Group {i + 1}')
    plt.xlabel('销量')
    plt.ylabel('成本加成定价')

plt.tight_layout()
plt.show()