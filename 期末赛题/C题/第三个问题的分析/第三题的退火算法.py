import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
import random
from simanneal import Annealer

# 定义文件路径
file_path = 'D:\\新建文件夹\\期末赛题\\期末赛题\\C题\\每日总销量_with_avg.xlsx'
output_file_path = 'D:\\新建文件夹\\期末赛题\\期末赛题\\C题\\优化结果.xlsx'

# 读取Excel数据
df = pd.read_excel(file_path)

# 数据预处理
df['损耗率（%）'] = df['损耗率（%）'] / 100  # 损耗率从百分比转换为小数

# 计算每个商品的最大补货量
max_restock = df.groupby('商品名称')['销量（千克）'].max().reset_index()
max_restock.columns = ['商品名称', '最大补货量']

# 计算平均销售单价、批发价格和损耗率
df_avg = df.groupby('商品名称')[['销售单价（元/千克）', '批发价格（元/千克）', '损耗率（%）']].mean().reset_index()

# 合并数据
merged_df = pd.merge(max_restock, df_avg, on='商品名称')


# 删除数据中的单位（万一数据中有非数字）
def clean_column_to_float(df, column):
    return pd.to_numeric(df[column], errors='coerce')


max_restock_values = clean_column_to_float(merged_df, '最大补货量').values
B = clean_column_to_float(merged_df, '批发价格（元/千克）').values
L = clean_column_to_float(merged_df, '损耗率（%）').values


# 定义目标函数
def total_revenue(params, data, B, L):
    y = params[:len(data)]  # 0-1变量
    R = params[len(data):2 * len(data)]  # 补货量
    P = params[2 * len(data):3 * len(data)]  # 定价

    selected_items = y.sum()

    # 约束条件处理
    if selected_items < 27 or selected_items > 33:
        return np.inf  # 违反约束条件，返回非常大的值，算法会尽量避免

    revenue = sum(y_i * (R_i * (P_i - B_i) - R_i * B_i * L_i)
                  for y_i, R_i, P_i, B_i, L_i
                  in zip(y, R, P, B, L))

    return -revenue  # 由于优化时默认最小化，因此返回负收益


# 设置优化变量的上下界（确保所有边界合理）
bounds = [(0, 1)] * len(merged_df) + [(2.5, mr if mr > 2.5 else 2.5) for mr in max_restock_values] + [(1.1 * b, 1.5 * b)
                                                                                                      for b in B]

# 使用微分进化算法进行优化
result = differential_evolution(total_revenue, bounds, args=(merged_df, B, L))

# 结果提取与展示
y_opt = result.x[:len(merged_df)]
R_opt = result.x[len(merged_df):2 * len(merged_df)]
P_opt = result.x[2 * len(merged_df):3 * len(merged_df)]

# 构造并展示最终优化结果
final_df = merged_df.copy()
final_df['是否补货'] = y_opt
final_df['补货量'] = R_opt
final_df['定价'] = P_opt

# 保存到Excel文件
final_df.to_excel(output_file_path, index=False)
print(f"优化结果已保存到 {output_file_path}")


# 使用模拟退火算法进行优化（备用方案）
class VegetablesOptimization(Annealer):
    def __init__(self, initial_state, data):
        self.data = data
        super(VegetablesOptimization, self).__init__(initial_state)

    def move(self):
        # 随机选择一个商品并修改其补货状态
        idx = random.randint(0, len(self.state) - 1)
        self.state[idx] = (self.state[idx] + 1) % 2  # flip 0-1

    def energy(self):
        return total_revenue(self.state, self.data, B, L)


# 初始状态：随机0-1变量
initial_state = [random.randint(0, 1) for _ in range(len(merged_df))]

annealer = VegetablesOptimization(initial_state, merged_df)
annealer.Tmax = 100.0
annealer.Tmin = 0.1
annealer.steps = 1000
annealer.updates = 100

state, e = annealer.anneal()

# 结果分析与展示
final_state_df = merged_df.copy()
final_state_df['是否补货'] = state

# 保存模拟退火优化结果到Excel文件
final_state_output_file_path = 'D:\\新建文件夹\\期末赛题\\期末赛题\\C题\\模拟退火优化结果.xlsx'
final_state_df.to_excel(final_state_output_file_path, index=False)
print(f"模拟退火优化结果已保存到 {final_state_output_file_path}")                                                                                                     