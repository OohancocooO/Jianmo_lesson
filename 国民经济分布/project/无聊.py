import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置matplotlib支持中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 'SimHei' 是一种支持中文的字体
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 数据
cities = [
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
]
x1 = [
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
]
x2 = [65.5, 67.3, 69.9, 74.9, 49.9, 48.6, 47.3, 46.1, 45.1, 48.5, 40.5, 43.9, 34.1]
x3 = [
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
    542.00,
]
# 圆圈大小
sizes = [population * 10 for population in x1]  # 放大10倍以便可视化
# 圆圈颜色
colors = []
for population in x1:
    if population > 1000:
        colors.append("#E4041C")  # 超大城市
    elif population > 500:
        colors.append("#007CBC")  # 特大城市
    elif population > 100:
        colors.append("#F9C114")  # 大城市
    elif population > 50:
        colors.append("#F59EC8")  # 中等城市
    else:
        colors.append("#3CD201")  # 小城市

# 绘制散点图
plt.figure(figsize=(14, 8))
scatter = plt.scatter(
    x3, x2, s=sizes, c=colors, alpha=0.6, edgecolors="w", linewidth=0.5
)

# 添加标签
for i, city in enumerate(cities):
    plt.text(x3[i], x2[i], city, fontsize=9, ha="right")

# 添加图例
from matplotlib.lines import Line2D

legend_elements = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="超大城市",
        markerfacecolor="#E4041C",
        markersize=15,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="特大城市",
        markerfacecolor="#007CBC",
        markersize=15,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="大城市",
        markerfacecolor="#F9C114",
        markersize=15,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="中等城市",
        markerfacecolor="#F59EC8",
        markersize=15,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="小城市",
        markerfacecolor="#3CD201",
        markersize=15,
    ),
]

plt.legend(handles=legend_elements, loc="upper left")

# 设置标题和标签
plt.title("2007年江苏省各地市经济指标散点图")
plt.xlabel("地区生产总值(GDP)(亿元)")
plt.ylabel("城镇化率(%)")

# 显示图表
plt.grid(True)
plt.show()
