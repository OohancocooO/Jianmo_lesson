import numpy as np  # 科学计算工具包
import matplotlib.pyplot as plt  # 画图工具包

# 设置matplotlib支持中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 定义SIS函数——根据SIS模型计算每天新增的易感人数和感染人数，返回累计人数
def SIS(sis,dt):
    S,I = sis          # 每天初始SI的数值
    dS = -(r*I*S)/N + u*I   # 易感者微分方程
    dI = r*I*S/N - u*I     # 感染者微分方程
    S = 0 if S+dS*dt<=0 else S+dS*dt    # 当天易感者人数
    I = N if I+dI*dt>=N else I+dI*dt    # 当天感染者人数
    return [S, I]

def calculate(func,sis,days):
    dt = 1
    t = np.arange(0,days,dt) # 设置时间步
    res=[]
    for itm in t:
        sis=func(sis,dt)      # 运行SI模型函数
        res.append(sis)      # 存储每天人数结果
    return np.array(res)

# 画图函数
def plot_graph(np_res):
    plt.figure(figsize=(10,6),dpi=300)
    plt.plot(np_res[:,0])
    plt.plot(np_res[:,1])
    plt.title("SIS模型")
    plt.xlabel("天数")
    plt.ylabel("人数")
    plt.legend(['易感者','感染者'])
    plt.text(10,4500,'N=%d \nI=%d \nr=%2.1f \nu=%2.1f' % (N,I,r,u), bbox={'facecolor': 'red', 'alpha': 0.4, 'pad': 8})
    final = [round(x) for x in result[-1]]  # 取整表示
    plt.text(90,final[0],final[0])
    plt.text(93,final[1],final[1])
    plt.show()

# 赋值绘图
N = 10000
I = 1
r = 0.6
u = 0.2
days = 100

sis= [N-I,    # 易感人数
       I]    # 感染人数

result = calculate(SIS,sis,days)
plot_graph(result)
print(result[-1])
