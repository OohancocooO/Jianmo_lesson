import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei' 是一种支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 氮元素数据：施肥量和生菜产量
fertilizer_N = np.array([0, 34, 67, 101, 135, 202, 259, 336, 404, 471])  # 千克每公顷
yield_data_N = np.array([15.18, 21.36, 25.72, 32.29, 34.04, 39.45, 43.15, 43.46, 40.83, 30.75])  # 吨每公顷

# 磷元素数据：施肥量和生菜产量
fertilizer_P = np.array([0, 24, 49,  73, 98, 147, 196, 245, 294, 342])  # 千克每公顷
yield_data_P = np.array([33.46, 32.47, 36.06, 37.96, 41.04, 40.09, 41.26, 42.17, 40.36, 42.73])  # 吨每公顷

# 钾元素数据：施肥量和生菜产量
fertilizer_K = np.array([0, 47, 93, 140, 186, 279, 372, 465, 558, 651])  # 千克每公顷
yield_data_K = np.array([18.98, 27.35, 34.86, 38.52, 38.44, 37.73, 38.43, 43.87, 42.77, 46.22])  # 吨每公顷

# 创建多元线性回归模型的数据集
X = np.column_stack((fertilizer_N, fertilizer_P, fertilizer_K))  # 特征矩阵
y = np.column_stack((yield_data_N, yield_data_P, yield_data_K))  # 目标矩阵

# 添加常数项到特征矩阵中，以便进行多元线性回归
X = sm.add_constant(X)

# 使用statsmodels进行多元线性回归分析
model = sm.OLS(y, X).fit()

# 打印回归模型的详细结果
print(model.summary())

# 可视化多元线性回归结果
# 注意：这里仅以氮元素的施肥量和生菜产量为例进行可视化
plt.scatter(fertilizer_N, yield_data_N, color='blue', label='氮元素')
plt.scatter(fertilizer_P, yield_data_P, color='green', label='磷元素')
plt.scatter(fertilizer_K, yield_data_K, color='red', label='钾元素')

# 绘制氮元素的回归线
plt.plot(fertilizer_N, model.predict(X)[::3], color='blue', label='氮元素回归线')

plt.xlabel('施肥量 (千克每公顷)')
plt.ylabel('土豆产量 (吨每公顷)')
plt.title('三种元素对生菜产量的多元线性回归分析')
plt.legend()
plt.show()