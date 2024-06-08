import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# 设置matplotlib支持中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 'SimHei' 是一种支持中文的字体
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号
# 加载数据
data = pd.read_csv(r'D:\Jianmo_lesson\国民经济分布\Dataset\dataset.txt', delim_whitespace=True)

# 数据预处理
features = data.iloc[:, 1:]
scaler = StandardScaler()
data_standardized = scaler.fit_transform(features)

# 使用肘部法确定最佳聚类数目
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_standardized)
    sse.append(kmeans.inertia_)

# 绘制肘部法图形
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.title('Elbow Method For Optimal k')
plt.show()

# 选择最佳聚类数目
k = 3

# 进行K-means聚类
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(data_standardized)
labels = kmeans.labels_

# 将聚类标签添加到原数据中
data['Cluster'] = labels

# PCA降维
pca = PCA(n_components=2)
data_2d = pca.fit_transform(data_standardized)

# 绘制聚类结果并标注城市名称
plt.figure(figsize=(10, 8))
scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis')
for i, txt in enumerate(data['地市']):
    plt.annotate(txt, (data_2d[i, 0], data_2d[i, 1]), textcoords="offset points", xytext=(0,10), ha='center')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-means Clustering Result')
plt.colorbar(scatter, label='Cluster')
plt.show()

# 查看聚类中心
centers = kmeans.cluster_centers_
centers = scaler.inverse_transform(centers)  # 将中心点还原到原始数据的尺度
centers_df = pd.DataFrame(centers, columns=features.columns)
print(centers_df)