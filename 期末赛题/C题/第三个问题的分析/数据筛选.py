import pandas as pd

# 读取Excel文件
file_path = 'D:\\新建文件夹\\期末赛题\\期末赛题\\C题\\6月24到6月30日的数据.xlsx'
xl = pd.ExcelFile(file_path)

# 假设我们要读取的是第一个sheet
df = xl.parse(sheet_name=0)

# 读取所需的列：销售日期、商品名称、销量、销售单价、批发价格、损耗率
df = df.iloc[:, [0, 1, 3, 4, 7, 8]]
df.columns = ['销售日期', '商品名称', '销量（千克）', '销售单价（元/千克）', '批发价格（元/千克）', '损耗率（%）']

# 将“销售日期”转换为日期类型
df['销售日期'] = pd.to_datetime(df['销售日期'])

# 计算每种商品的总销量，筛选出总销量最高的49种蔬菜
top_49_items = df.groupby('商品名称')['销量（千克）'].sum().nlargest(49).index

# 从表中筛选出这49种蔬菜的数据
filtered_df = df[df['商品名称'].isin(top_49_items)]

# 计算每日总销量
daily_sales = filtered_df.groupby(['销售日期', '商品名称'])['销量（千克）'].sum().reset_index()

# 计算每种商品的平均销售单价、批发价格和损耗率
avg_prices = filtered_df.groupby('商品名称')[['销售单价（元/千克）', '批发价格（元/千克）', '损耗率（%）']].mean().reset_index()

# 将平均价格和损耗率添加到daily_sales的每种商品后面
daily_sales = daily_sales.merge(avg_prices, on='商品名称', how='left')

# 打印或保存每日总销量
print(daily_sales.head(100))  # 仅展示前100行数据

# 你可以将结果保存到一个新的Excel文件
output_file = 'D:\\新建文件夹\\期末赛题\\期末赛题\\C题\\每日总销量_with_avg.xlsx'
daily_sales.to_excel(output_file, index=False)