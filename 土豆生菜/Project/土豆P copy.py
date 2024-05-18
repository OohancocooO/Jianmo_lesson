import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# 土豆数据


P_data = np.array([0, 24, 49, 73, 98, 147, 196, 245, 294, 342])
P_yields = np.array(
    [33.46, 32.47, 36.06, 37.96, 41.04, 40.09, 41.26, 42.17, 40.36, 42.73]
)

K_data = np.array([0, 47, 93, 140, 186, 279, 372, 465, 558, 651])
K_yields = np.array(
    [18.98, 27.35, 34.86, 38.52, 38.44, 37.73, 38.43, 43.87, 42.77, 46.22]
)


def poly_func(x, beta1, beta2, beta3, beta4):
    return beta1 * x**3 + beta2 * x**2 + beta3 * x + beta4


params, _ = curve_fit(poly_func, P_data, P_yields)

y_pred = poly_func(P_data, *params)
rss = np.sum((P_yields - y_pred) ** 2)
tss = np.sum((P_yields - np.mean(P_yields)) ** 2)
r2 = 1 - rss / tss

plt.plot(P_data, P_yields, "bo", label="p data")
plt.plot(
    P_data,
    y_pred,
    "r-",
    label="fit: beta1=%5.3f, beta2=%5.3f, beta3=%5.3f, beta4=%5.3fR^2=%5.3f"
    % (*params, r2),
)

plt.legend()
equation = (
    f"y = {params[0]:.4f}x^3 + {params[1]:.4f}x^2 + {params[2]:.4f}x + {params[3]:.4f}"
)

plt.text(
    0.55,
    0.1,
    equation,
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment="top",
)

plt.show()

print("多项式函数拟合 R^2:", r2)
