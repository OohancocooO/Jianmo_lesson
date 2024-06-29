import pandas as pd
import numpy as np
from scipy.stats import pearsonr, jarque_bera, t
import statsmodels.api as sm

# 读取Excel文件
file_path = r"C:\Users\86184\Desktop\perason.xlsx"
df = pd.read_excel(file_path, sheet_name='Sheet1', usecols="B:M", skiprows=2, nrows=1085)

# 将NaN和inf值替换为0
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# 计算各列之间的相关系数以及P值
corr_matrix = df.corr()
p_values_matrix = pd.DataFrame(index=df.columns, columns=df.columns)

for i in range(len(df.columns)):
    for j in range(len(df.columns)):
        _, p_value = pearsonr(df.iloc[:, i], df.iloc[:, j])
        p_values_matrix.iloc[i, j] = p_value

# 用循环检验所有列的数据的正态分布性
n_c = df.shape[1]  # number of columns 数据的列数
H = np.zeros(n_c)  # 初始化节省时间和消耗
P_val = np.zeros(n_c)  # 计算所得检验p值

for i in range(n_c):
    h, p = jarque_bera(df.iloc[:, i])
    H[i] = h
    P_val[i] = p

print("H:", H)
print("P_val:", P_val)

# 检验相关系数
r = 0.5
n = int(input("请输入样本数量："))
alpha = float(input("请输入显著性检验判断值："))

t_value = r * np.sqrt((n - 2) / (1 - r**2))  # n为样本数量
p_value = (1 - t.cdf(t_value, n - 2)) * 2  # 此时的t为输入n后求得的t值，p即为求得的检验值

print("检验相关系数得到的p值为：", p_value)
if p_value < alpha:
    print("拒绝原假设，相关系数显著不等于0")
else:
    print("接受原假设，相关系数显著等于0")

