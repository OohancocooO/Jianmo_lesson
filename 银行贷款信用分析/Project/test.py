import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
import seaborn as sns
import matplotlib.pyplot as plt


# 创建数据框
data = {
    "信用好坏": ["已履行还贷责任"] * 5 + ["未履行还贷责任"] * 5,
    "X1": [23, 34, 42, 39, 35, 37, 29, 32, 28, 26],
    "X2": [1, 1, 2, 1, 1, 1, 1, 2, 2, 1],
    "X3": [7, 17, 7, 19, 9, 1, 13, 11, 2, 4],
    "X4": [2, 3, 23, 5, 1, 3, 1, 6, 3, 3],
    "X5": [31, 59, 41, 48, 34, 24, 42, 75, 23, 27],
    "X6": [6.6, 8, 4.6, 13.1, 5, 15.1, 7.4, 23.3, 6.4, 10.5],
    "X7": [0.34, 1.81, 0.94, 1.93, 0.4, 1.1, 1.46, 7.76, 0.19, 2.47],
    "X8": [1.71, 2.91, 0.94, 4.36, 1.3, 1.82, 1.65, 7.22, 1.29, 0.36],
}

df = pd.DataFrame(data)

# 新客户数据
new_customer = np.array([53, 1, 9, 18, 50, 1.20, 2.02, 3.58])

# 去除高度相关的变量，假设我们选择去除X7
df_reduced = df.drop(columns=["X7"])

# 更新新客户数据
new_customer_reduced = np.array([53, 1, 9, 18, 50, 1.20, 3.58])

# 计算相关系数矩阵
corr_matrix = df.drop(columns="信用好坏").corr()

# 可视化相关系数矩阵
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.show()

# 根据相关系数矩阵，我们可以选择去除一个或多个高度相关的变量。

# 计算类别均值和协方差矩阵
grouped_reduced = df_reduced.groupby("信用好坏")
means_reduced = grouped_reduced.mean()
cov_matrix_reduced = np.cov(df_reduced.drop(columns="信用好坏").values, rowvar=False)
inv_cov_matrix_reduced = np.linalg.inv(cov_matrix_reduced)

# 计算到各类别均值的马氏距离
distances_reduced = {}
for label, mean in means_reduced.iterrows():
    dist = distance.mahalanobis(new_customer_reduced, mean, inv_cov_matrix_reduced)
    distances_reduced[label] = dist

# 找出最小距离的类别
mahalanobis_result_reduced = min(distances_reduced, key=distances_reduced.get)
print(f"马氏距离判别结果: {mahalanobis_result_reduced}")

# 准备降维后的训练数据
X_reduced = df_reduced.drop(columns="信用好坏").values
y_reduced = (
    df_reduced["信用好坏"].apply(lambda x: 1 if x == "已履行还贷责任" else 0).values
)

# 线性判别分析
lda_reduced = LinearDiscriminantAnalysis()
lda_reduced.fit(X_reduced, y_reduced)
lda_result_reduced = lda_reduced.predict([new_customer_reduced])
lda_result_label_reduced = (
    "已履行还贷责任" if lda_result_reduced == 1 else "未履行还贷责任"
)
print(f"线性判别分析结果: {lda_result_label_reduced}")

# 二次判别分析
qda_reduced = QuadraticDiscriminantAnalysis()
qda_reduced.fit(X_reduced, y_reduced)
qda_result_reduced = qda_reduced.predict([new_customer_reduced])
qda_result_label_reduced = (
    "已履行还贷责任" if qda_result_reduced == 1 else "未履行还贷责任"
)
print(f"二次判别分析结果: {qda_result_label_reduced}")
