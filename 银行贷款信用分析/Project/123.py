import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)

# 样本数据
data = np.array(
    [
        [23, 1, 7, 2, 31, 6.6, 0.34, 1.71],
        [34, 1, 17, 3, 59, 8, 1.81, 2.91],
        [42, 2, 7, 23, 41, 4.6, 0.94, 0.94],
        [39, 1, 19, 5, 48, 13.1, 1.93, 4.36],
        [35, 1, 9, 1, 34, 5, 0.4, 1.3],
        [37, 1, 1, 3, 24, 15.1, 1.1, 1.82],
        [29, 1, 13, 1, 42, 7.4, 1.46, 1.65],
        [32, 2, 11, 6, 75, 23.3, 7.76, 7.22],
        [28, 2, 2, 3, 23, 6.4, 0.19, 1.29],
        [26, 1, 4, 3, 27, 10.5, 2.47, 0.36],
    ]
)

labels = np.array(
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
)  # 1表示已履行还贷责任，0表示未履行还贷责任

# 待判别客户数据
new_customer = np.array([53, 1, 9, 18, 50, 1.20, 2.02, 3.58])


# 马氏距离判别法
def mahalanobis_distance(new_customer, data, labels):
    good_customers = data[labels == 1]
    bad_customers = data[labels == 0]

    mean_good = np.mean(good_customers, axis=0)
    mean_bad = np.mean(bad_customers, axis=0)

    cov_good = np.cov(good_customers, rowvar=False)
    cov_bad = np.cov(bad_customers, rowvar=False)

    inv_cov_good = np.linalg.pinv(cov_good)
    inv_cov_bad = np.linalg.pinv(cov_bad)

    dist_good = mahalanobis(new_customer, mean_good, inv_cov_good)
    dist_bad = mahalanobis(new_customer, mean_bad, inv_cov_bad)

    return 1 if dist_good < dist_bad else 0


# 线性判别法
def linear_discriminant(new_customer, data, labels):
    lda = LinearDiscriminantAnalysis()
    lda.fit(data, labels)
    return lda.predict([new_customer])[0]


# 二次判别法
def quadratic_discriminant(new_customer, data, labels):
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(data, labels)
    return qda.predict([new_customer])[0]


# 结果输出
print(
    "马氏距离判别法结果:",
    (
        "已履行还贷责任"
        if mahalanobis_distance(new_customer, data, labels) == 1
        else "未履行还贷责任"
    ),
)
print(
    "线性判别法结果:",
    (
        "已履行还贷责任"
        if linear_discriminant(new_customer, data, labels) == 1
        else "未履行还贷责任"
    ),
)
print(
    "二次判别法结果:",
    (
        "已履行还贷责任"
        if quadratic_discriminant(new_customer, data, labels) == 1
        else "未履行还贷责任"
    ),
)
