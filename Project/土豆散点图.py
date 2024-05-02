import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial


# 数据
N_fertilizer = np.array([0, 34, 67, 101, 135, 202, 259, 336, 404, 471])
N_yield = np.array(
    [15.18, 21.36, 25.72, 32.29, 34.03, 39.45, 43.15, 43.46, 40.83, 30.75]
)

P_fertilizer = np.array([0, 24, 49, 73, 98, 147, 196, 245, 294, 342])
P_yield = np.array(
    [33.46, 32.47, 36.06, 37.96, 41.04, 40.09, 41.26, 42.17, 40.36, 42.73]
)

K_fertilizer = np.array([0, 47, 93, 140, 186, 279, 372, 465, 558, 651])
K_yield = np.array(
    [18.98, 27.35, 34.86, 38.52, 38.44, 37.73, 38.43, 43.87, 42.77, 46.22]
)


def descriptive_stats(fertilizer, yield_):
    return {
        "mean_fertilizer": np.mean(fertilizer),
        "std_fertilizer": np.std(fertilizer, ddof=1),
        "min_fertilizer": np.min(fertilizer),
        "max_fertilizer": np.max(fertilizer),
        "mean_yield": np.mean(yield_),
        "std_yield": np.std(yield_, ddof=1),
        "min_yield": np.min(yield_),
        "max_yield": np.max(yield_),
    }


# 绘图函数
def plot_fertilizer_yield(fertilizer, yield_, nutrient_name):
    # 拟合多项式回归（这里为了演示，我们暂时使用2阶多项式）
    coefs = Polynomial.fit(fertilizer, yield_, 2).convert().coef
    p = np.poly1d(coefs[::-1])  # 多项式对象

    # 生成趋势线的数据
    x_line = np.linspace(min(fertilizer), max(fertilizer), 100)
    y_line = p(x_line)

    plt.figure(figsize=(8, 5))
    plt.scatter(fertilizer, yield_, label=f"{nutrient_name} Fertilizer vs Yield")
    plt.plot(x_line, y_line, "r-", label="Trend Line")
    plt.xlabel(f"{nutrient_name} Fertilizer Quantity")
    plt.ylabel("Yield")

    # 组装方程字符串
    equation_text = f"${p.c[0]:.2f} + {p.c[1]:.2f}x + {p.c[2]:.2f}x^2$"
    # 将文本调整到图的右侧
    plt.text(
        max(fertilizer) * 0.95,
        max(yield_) * 0.85,
        f"Eq: {equation_text}",
        fontsize=12,
        color="blue",
        ha="right",
        va="center",
    )

    plt.title(f"{nutrient_name} Fertilizer vs Yield with Trend Line")
    plt.legend()
    plt.show()


# 绘制N、P、K的散点图和趋势线
plot_fertilizer_yield(N_fertilizer, N_yield, "N")
plot_fertilizer_yield(P_fertilizer, P_yield, "P")
plot_fertilizer_yield(K_fertilizer, K_yield, "K")
