import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 氮、磷、钾元素的数据
fertilizer_N = np.array([0, 34, 67, 101, 135, 202, 259, 336, 404, 471])
yield_data_N = np.array([15.18, 21.36, 25.72, 32.29, 34.04, 39.45, 43.15, 43.46, 40.83, 30.75])
fertilizer_P = np.array([0, 24, 49,  73, 98, 147, 196, 245, 294, 342])
yield_data_P = np.array([33.46, 32.47, 36.06, 37.96, 41.04, 40.09, 41.26, 42.17, 40.36, 42.73])
fertilizer_K = np.array([0, 47, 93, 140, 186, 279, 372, 465, 558, 651])
yield_data_K = np.array([18.98, 27.35, 34.86, 38.52, 38.44, 37.73, 38.43, 43.87, 42.77, 46.22])

# 创建一个DataFrame来存储所有元素的数据
data = pd.DataFrame({
    '施肥量_N': fertilizer_N,
    '产量_N': yield_data_N,
    '施肥量_P': fertilizer_P,
    '产量_P': yield_data_P,
    '施肥量_K': fertilizer_K,
    '产量_K': yield_data_K
})

# 为每种元素的施肥量添加多项式特征
data['施肥量_N_平方'] = data['施肥量_N'] ** 2
data['施肥量_P_平方'] = data['施肥量_P'] ** 2
data['施肥量_K_平方'] = data['施肥量_K'] ** 2

# 构建三个多项式回归模型
model_N = ols('产量_N ~ 施肥量_N + 施肥量_N_平方', data).fit()
model_P = ols('产量_P ~ 施肥量_P + 施肥量_P_平方', data).fit()
model_K = ols('产量_K ~ 施肥量_K + 施肥量_K_平方', data).fit()

# 创建一个图表，包含三个子图
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# 绘制氮元素的数据点和回归线
axs[0].scatter(data['施肥量_N'], data['产量_N'], color='blue', label='数据点')
x_fit_N = np.linspace(min(data['施肥量_N']), max(data['施肥量_N']), 400)
y_fit_N = model_N.predict(pd.DataFrame({
    '施肥量_N': x_fit_N,
    '施肥量_N_平方': x_fit_N ** 2
}))
axs[0].plot(x_fit_N, y_fit_N, color='red', label='二次拟合')
axs[0].set_title('氮元素对土豆产量的影响')
axs[0].legend()

# 绘制磷元素的数据点和回归线
axs[1].scatter(data['施肥量_P'], data['产量_P'], color='green', label='数据点')
x_fit_P = np.linspace(min(data['施肥量_P']), max(data['施肥量_P']), 400)
y_fit_P = model_P.predict(pd.DataFrame({
    '施肥量_P': x_fit_P,
    '施肥量_P_平方': x_fit_P ** 2
}))
axs[1].plot(x_fit_P, y_fit_P, color='red', label='二次拟合')
axs[1].set_title('磷元素对土豆产量的影响')
axs[1].legend()

# 绘制钾元素的数据点和回归线
axs[2].scatter(data['施肥量_K'], data['产量_K'], color='purple', label='数据点')
x_fit_K = np.linspace(min(data['施肥量_K']), max(data['施肥量_K']), 400)
y_fit_K = model_K.predict(pd.DataFrame({
    '施肥量_K': x_fit_K,
    '施肥量_K_平方': x_fit_K ** 2
}))
axs[2].plot(x_fit_K, y_fit_K, color='red', label='二次拟合')
axs[2].set_title('钾元素对土豆产量的影响')
axs[2].legend()

# 设置子图间距
plt.tight_layout()

# 显示图表
plt.show()