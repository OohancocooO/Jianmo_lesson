import numpy as np
import matplotlib.pyplot as plt

# 数据
fertilizer = np.array([0, 47, 93, 140, 186, 279, 372, 465, 558, 651])
yield_ = np.array(
    [15.75, 16.76, 16.89, 16.24, 17.56, 19.20, 17.97, 15.84, 20.11, 19.40]
)

# 回归系数
a = -7.189e-7
b = 0.0051
c = 16.2329

# 置信区间
a_ci = 2.197e-7
b_ci = 0.0195
c_ci = 1.7979

# 生成x值
x = np.linspace(0, 685, 500)

# 回归曲线
y = a * x**2 + b * x + c

# 置信区间上下界
y_lower = (a - a_ci) * x**2 + (b - b_ci) * x + (c - c_ci)
y_upper = (a + a_ci) * x**2 + (b + b_ci) * x + (c + c_ci)

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制回归曲线
plt.plot(fertilizer, yield_, "o", label="Data points")
plt.plot(x, y, "b-", label="Regression curve")

# 绘制置信区间
plt.plot(x, y_lower, color="#FF8899", label="95% Confidence interval")
plt.plot(x, y_upper, color="#FF8899")
plt.fill_between(x, y_lower, y_upper, color="#FFDD88", alpha=0.5)

# 添加标签和标题
plt.xlabel("Fertilizer amount")
plt.ylabel("Yield")
plt.title("Regression Curve with 95% Confidence Interval")
plt.legend()

# 显示图形
plt.show()
