import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from matplotlib.font_manager import FontProperties

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei' 是一种支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 施肥量和生菜产量的数据
fertilizer = np.array([0, 28, 56, 84, 112, 168, 224, 280, 336, 392])  # 千克每公顷
yield_data = np.array([11.02, 12.70, 14.56, 16.27, 17.75, 22.59, 21.63, 19.34, 16.12, 14.11])  # 吨每公顷

# 将数据转换为DataFrame
data = pd.DataFrame({
    '施肥量_千克_公顷': fertilizer,
    '生菜产量_吨_公顷': yield_data
})

# 为施肥量添加多项式特征，这里以二次多项式为例
data['施肥量_平方'] = data['施肥量_千克_公顷'] ** 2

# 构建多项式回归模型，使用'生菜产量_吨 ~ 施肥量_千克_公顷 + 施肥量_平方'来表示包含二次项的多项式回归
model = ols('生菜产量_吨_公顷 ~ 施肥量_千克_公顷 + 施肥量_平方', data).fit()

# 打印模型的摘要信息
print(model.summary())

# 获取回归模型的预测值
predictions = model.predict(data)

# 绘制原始数据点
plt.scatter(data['施肥量_千克_公顷'], data['生菜产量_吨_公顷'], color='blue', label='数据点')

# 生成用于绘制多项式回归线的点
x_fit = np.linspace(min(data['施肥量_千克_公顷']), max(data['施肥量_千克_公顷']), 400)
x_fit_squared = x_fit ** 2
y_fit = model.params['Intercept'] + model.params['施肥量_千克_公顷'] * x_fit + model.params['施肥量_平方'] * x_fit_squared

# 绘制多项式回归线
plt.plot(x_fit, y_fit, color='red', label='二次拟合')

# 添加图例
plt.legend()

# 添加标题和轴标签
plt.title('生菜产量与施肥量 - 氮元素')
plt.xlabel('施肥量 (千克/公顷)')
plt.ylabel('生菜产量 (吨/公顷)')

# 显示图表
plt.show()
