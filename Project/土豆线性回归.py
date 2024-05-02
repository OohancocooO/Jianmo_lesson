from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import numpy as np

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


# 多项式回归分析函数
def polynomial_regression_analysis(fertilizer, yield_, degree=2):
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(fertilizer.reshape(-1, 1))
    model = LinearRegression()
    model.fit(X_poly, yield_)
    y_pred = model.predict(X_poly)

    # 计算R^2分数
    r2 = r2_score(yield_, y_pred)

    return {"coefs": model.coef_, "intercept": model.intercept_, "r2_score": r2}


# 对N、P、K进行多项式回归分析
regression_results_N = polynomial_regression_analysis(N_fertilizer, N_yield)
regression_results_P = polynomial_regression_analysis(P_fertilizer, P_yield)
regression_results_K = polynomial_regression_analysis(K_fertilizer, K_yield)

print("N Polynomial Regression Results:", regression_results_N)
print("P Polynomial Regression Results:", regression_results_P)
print("K Polynomial Regression Results:", regression_results_K)
