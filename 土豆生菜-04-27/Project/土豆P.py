import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# 土豆数据
N_data = np.array([0, 34, 67, 101, 135, 202, 259, 336, 404, 471])
N_yields = np.array(
    [15.18, 21.36, 25.72, 32.29, 34.03, 39.45, 43.15, 43.46, 40.83, 30.75]
)

P_data = np.array([0, 24, 49, 73, 98, 147, 196, 245, 294, 342])
P_yields = np.array(
    [33.46, 32.47, 36.06, 37.96, 41.04, 40.09, 41.26, 42.17, 40.36, 42.73]
)

K_data = np.array([0, 47, 93, 140, 186, 279, 372, 465, 558, 651])
K_yields = np.array(
    [18.98, 27.35, 34.86, 38.52, 38.44, 37.73, 38.43, 43.87, 42.77, 46.22]
)


def log_func(x, beta1, beta2, beta3):
    return beta1 + beta2 * np.log(x + beta3)


params, _ = curve_fit(log_func, P_data, P_yields)

y_pred = log_func(P_data, *params)
rss = np.sum((P_yields - y_pred) ** 2)
tss = np.sum((P_yields - np.mean(P_yields)) ** 2)
r2 = 1 - rss / tss

plt.plot(P_data, P_yields, "bo", label="P data")
plt.plot(
    P_data,
    y_pred,
    "r-",
    label="fit: beta1=%5.3f, beta2=%5.3f, beta3=%5.3f, R^2=%5.3f" % (*params, r2),
)
plt.legend()
plt.show()

print("对数函数拟合 R^2:", r2)
