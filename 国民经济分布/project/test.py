import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

# 设置matplotlib支持中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 'SimHei' 是一种支持中文的字体
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 数据准备
data = {
    "City": [
        "苏州",
        "无锡",
        "常州",
        "南京",
        "镇江",
        "南通",
        "扬州",
        "盐城",
        "泰州",
        "徐州",
        "连云港",
        "淮安",
        "宿迁",
    ],
    "x1": [
        624.43,
        461.74,
        357.38,
        617.17,
        268.78,
        766.13,
        450.68,
        500.7,
        450.5,
        940.95,
        482.23,
        539.09,
        531.53,
    ],
    "x2": [
        65.5,
        67.3,
        69.9,
        74.9,
        49.9,
        48.6,
        47.3,
        46.1,
        45.1,
        48.5,
        40.5,
        43.9,
        34.1,
    ],
    "x3": [
        5700.85,
        3858.54,
        1881.28,
        2833.11,
        1025.24,
        2119.35,
        1811.35,
        1201.86,
        1679.56,
        1876.86,
        618.16,
        765.23,
        542.0,
    ],
    "x4": [7.4, 9.1, 18.6, 11.0, 24.5, 35.1, 35.3, 32.3, 36.0, 36.2, 38.4, 32.4, 32.0],
    "x5": [
        1704.27,
        1180.74,
        748.89,
        1443.4,
        363.73,
        633.94,
        438.35,
        347.73,
        769.59,
        409.56,
        394.91,
        470.06,
        256.18,
    ],
    "x6": [
        1250.05,
        1134.75,
        610.85,
        1380.46,
        331.36,
        736.54,
        418.9,
        321.07,
        543.01,
        249.08,
        269.4,
        433.74,
        158.87,
    ],
    "x7": [
        21260,
        20899,
        19235,
        20317,
        17765,
        16451,
        15057,
        14940,
        14875,
        13254,
        12164,
        13857,
        9468,
    ],
    "x8": [
        37.9,
        39.8,
        38.5,
        35.3,
        38.7,
        38.5,
        37.9,
        43.1,
        34.9,
        38.9,
        38.9,
        38.5,
        42.4,
    ],
    "x9": [
        10475,
        10026,
        9033,
        8020,
        7668,
        6905,
        6586,
        6469,
        5534,
        4823,
        5010,
        6092,
        4783,
    ],
    "x10": [35.7, 37.6, 38, 37.4, 39.4, 37.9, 38.9, 38.1, 39.9, 43.7, 43.2, 41.7, 46],
}

df = pd.DataFrame(data)

# 提取特征值
X = df.iloc[:, 1:]

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用肘部法则确定聚类数
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

# 绘制肘部图
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), sse, marker="o")
plt.xlabel("聚类数")
plt.ylabel("SSE")
plt.title("肘部法则确定最佳聚类数")
plt.show()

# 选定聚类数（假设我们选择3类）
kmeans = KMeans(n_clusters=3, random_state=0)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# 聚类结果可视化
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="x3", y="x7", hue="Cluster", palette="viridis", s=100)

# 标注城市名
for i in range(df.shape[0]):
    plt.text(df["x3"][i], df["x7"][i], df["City"][i], fontsize=10, ha="right")


plt.title("聚类结果")
plt.xlabel("地区生产总值(GDP)(亿元)")
plt.ylabel("城市居民人均可支配收入(元)")
plt.show()

print(df)
