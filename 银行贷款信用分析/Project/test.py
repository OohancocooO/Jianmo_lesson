import numpy as np
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from scipy.linalg import inv

# 数据
data = np.array(
    [
        [23, 7, 2, 31, 6.6, 0.34, 1.71],
        [34, 17, 3, 59, 8, 1.81, 2.91],
        [42, 7, 23, 41, 4.6, 0.94, 0.94],
        [39, 19, 5, 48, 13.1, 1.93, 4.36],
        [35, 9, 1, 34, 5, 0.4, 1.3],
        [37, 1, 3, 24, 15.1, 1.1, 1.82],
        [29, 13, 1, 42, 7.4, 1.46, 1.65],
        [32, 11, 6, 75, 23.3, 7.76, 7.22],
        [28, 2, 3, 23, 6.4, 0.19, 1.29],
        [26, 4, 3, 27, 10.5, 2.47, 0.36],
    ]
)
labels = ["已履行还贷责任"] * 5 + ["未履行还贷责任"] * 5

# 新客户数据
new_customer = np.array([53, 1, 9, 18, 50, 1.20, 2.02, 3.58])

# 马氏距离判别
X_good = data[:5, :]
X_bad = data[5:, :]
mean_good = X_good.mean(axis=0)
mean_bad = X_bad.mean(axis=0)
cov_good = np.cov(X_good, rowvar=False)
cov_bad = np.cov(X_bad, rowvar=False)

# 为协方差矩阵增加微小的正则项以避免奇异性
regularization_term = 1e-6
cov_good += np.eye(cov_good.shape[0]) * regularization_term
cov_bad += np.eye(cov_bad.shape[0]) * regularization_term
mahalanobis_dist_good = np.dot(
    np.dot((new_customer[:-1] - mean_good), inv(cov_good)),
    (new_customer[:-1] - mean_good).T,
)
mahalanobis_dist_bad = np.dot(
    np.dot((new_customer[:-1] - mean_bad), inv(cov_bad)),
    (new_customer[:-1] - mean_bad).T,
)
# LDA模型
lda = LinearDiscriminantAnalysis()
lda.fit(data, labels)
prediction_lda = lda.predict([new_customer[:-1]])

# QDA模型
qda = QuadraticDiscriminantAnalysis()
qda.fit(data, labels)
prediction_qda = qda.predict([new_customer[:-1]])

# 输出结果
print("马氏距离（已履行还贷责任）:", mahalanobis_dist_good)
print("马氏距离（未履行还贷责任）:", mahalanobis_dist_bad)
print("LDA预测结果:", prediction_lda[0])
print("QDA预测结果:", prediction_qda[0])
