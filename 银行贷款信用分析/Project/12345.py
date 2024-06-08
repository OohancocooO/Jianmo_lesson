import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 设置matplotlib支持中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 'SimHei' 是一种支持中文的字体
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 加载数据
data = {
    "当前信用好坏": [
        "已履行还贷责任",
        "已履行还贷责任",
        "已履行还贷责任",
        "已履行还贷责任",
        "已履行还贷责任",
        "未履行还贷责任",
        "未履行还贷责任",
        "未履行还贷责任",
        "未履行还贷责任",
        "未履行还贷责任",
    ],
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

# 准备数据
X = df.drop("当前信用好坏", axis=1)
y = df["当前信用好坏"].map({"已履行还贷责任": 1, "未履行还贷责任": 0})  # 映射为二进制值

# 拆分数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=2024
)

# 初始化并训练LDA模型
lda = LDA(n_components=1)
lda.fit(X_train, y_train)

# 使用LDA模型进行预测
y_pred = lda.predict(X_test)

# 输出分类报告和混淆矩阵
report = classification_report(y_test, y_pred)
matrix = confusion_matrix(y_test, y_pred)

print(report)
print(matrix)

# 可视化
X_lda = lda.transform(X)

plt.figure(figsize=(10, 6))
colors = ["r", "b"]
markers = ["o", "s"]

for label, color, marker in zip(np.unique(y), colors, markers):
    plt.scatter(
        X_lda[y == label],
        np.zeros_like(X_lda[y == label]),
        color=color,
        label="已履行还贷责任" if label == 1 else "未履行还贷责任",
        marker=marker,
    )

plt.title("LDA: 1 component projection of the dataset")
plt.xlabel("LDA component 1")
plt.legend()
plt.show()

# 使用训练好的模型预测新客户的信用情况
new_customer = np.array([[53, 1, 9, 18, 50, 1.20, 2.02, 3.58]])
new_prediction = lda.predict(new_customer)

print(
    f"新客户的信用预测结果: {'已履行还贷责任' if new_prediction[0] == 1 else '未履行还贷责任'}"
)
