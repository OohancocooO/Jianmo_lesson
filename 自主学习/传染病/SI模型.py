import numpy as np  # 科学计算工具包
import matplotlib.pyplot as plt  # 画图工具包

# 设置matplotlib支持中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# 定义SI函数——根据SI模型计算每天新增的易感人数和感染人数，返回累计人数
def SI(si, dt):
    S, I = si  # 每天初始SI的数值
    dS = -(r * I * S) / N  # 易感者微分方程
    dI = r * I * S / N  # 感染者微分方程
    S = 0 if S + dS * dt <= 0 else S + dS * dt  # 当天易感者人数
    I = N if I + dI * dt >= N else I + dI * dt  # 当天感染者人数
    return [S, I]


def calculate(func, si, days):
    dt = 1
    t = np.arange(0, days, dt)  # 设置时间步
    res = []
    for itm in t:
        si = func(si, dt)  # 运行SI模型函数
        res.append(si)  # 存储每天人数结果
    return np.array(res)


# 画图函数
def plot_graph(np_res):
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(np_res[:, 0])
    plt.plot(np_res[:, 1])
    plt.title("SI模型")
    plt.xlabel("天数")
    plt.ylabel("人数")
    plt.legend(['易感者', '感染者'])
    plt.show()


N = 10000
I = 1
r = 1
days = 50
si = [N - I,  # 易感人数
      I]  # 感染人数

result = calculate(SI, si, days)
plot_graph(result)
