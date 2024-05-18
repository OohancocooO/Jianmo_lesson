import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import statsmodels.api as sm

# 数据
fertilizer_N = np.array([0, 28, 56, 84, 112, 168, 224, 280, 336, 392])
yield_N = np.array(
    [11.02, 12.70, 14.56, 16.27, 17.75, 22.59, 21.63, 19.34, 16.12, 14.11]
)
fertilizer_P = np.array([0, 49, 98, 147, 196, 294, 391, 489, 587, 685])
yield_P = np.array(
    [6.39, 9.48, 12.46, 14.33, 17.10, 21.49, 22.46, 21.34, 22.07, 24.53]
)
fertilizer_K = np.array([0, 47, 93, 140, 186, 279, 372, 465, 558, 651])
yield_K = np.array(
    [15.75, 16.76, 16.89, 16.24, 17.56, 19.20, 17.97, 15.84, 20.11, 19.40]
)


# 拟合函数
def quadratic(x, a, b, c):
    return a * x**2 + b * x + c


# 拟合数据
def fit_and_plot(fertilizer, yield_, element):
    params, covariance = curve_fit(quadratic, fertilizer, yield_)
    errors = np.sqrt(np.diag(covariance))

    # 生成回归系数和置信区间
    ci = 1.96 * errors

    # 计算显著性指标
    X = np.vstack((np.ones(len(fertilizer)), fertilizer, fertilizer**2)).T
    model = sm.OLS(yield_, X).fit()
    p_values = model.pvalues

    # 生成拟合数据
    x_fit = np.linspace(fertilizer.min(), fertilizer.max(), 100)
    y_fit = quadratic(x_fit, *params)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.scatter(fertilizer, yield_, label="Data")
    plt.plot(x_fit, y_fit, label="Fitted quadratic curve", color="red")
    plt.xlabel("Fertilizer Amount")
    plt.ylabel("Yield")
    plt.title(f"Fertilizer vs. Yield for {element}")
    plt.legend()
    equation = f"y = {params[0]:.4f}x^2 + {params[1]:.4f}x + {params[2]:.4f}"
    plt.text(
        0.55,
        0.1,
        equation,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
    )
    plt.show()

    # 输出回归系数和置信区间
    print(f"--- {element} ---")
    print(f"Coefficients: {params}")
    print(f"95% confidence intervals: {ci}")
    print(f"P-values: {p_values}")
    print(f"R-squared: {model.rsquared}\n")


# 分别拟合和绘制N, P, K的图
fit_and_plot(fertilizer_N, yield_N, "Nitrogen (N)")
fit_and_plot(fertilizer_P, yield_P, "Phosphorus (P)")
fit_and_plot(fertilizer_K, yield_K, "Potassium (K)")
