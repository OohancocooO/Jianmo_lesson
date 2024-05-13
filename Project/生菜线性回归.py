from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import numpy as np

# 数据
N_fertilizer = np.array([0, 28, 56, 84, 112, 168, 224, 280, 336, 392])
N_yield = np.array(
    [11.02, 12.07, 14.56, 16.27, 17.75, 22.59, 21.63, 19.34, 16.12, 14.11]
)

P_fertilizer = np.array([0,49,98,147,196,294,391,489,587,685])
P_yield = np.array(
    [6.39,9.48,12.46,14.33,17.1,21.94,22.64,21.34,22.07,24.53]
)

K_fertilizer = np.array([0,47,93,140,186,279,372,465,558,651])
K_yield = np.array(
    [15.75,16.76,16.89,16.24,17.56,19.2,17.97,15.84,20.11,19.4]
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
