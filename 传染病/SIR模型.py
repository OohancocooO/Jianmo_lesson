import numpy as np  # 科学计算工具包
import matplotlib.pyplot as plt  # 画图工具包

# 设置matplotlib支持中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 定义SIR函数——根据SIR模型计算每天新增的易感人数，感染人数，康复人数，返回累计人数
def SIR(sir,dt):
    S,I,R = sir             # 每天初始SIR的数值
    dS = -(r*I*S)/N         # 易感者微分方程
    dI = r*I*S/N - u*I      # 感染者微分方程
    dR = u*I                # 康复者微分方程
    S = 0 if S+dS*dt<=0 else S+dS*dt                        # 当天易感者人数
    I = N if I+dI*dt>=N else (0 if I+dI*dt<=0 else I+dI*dt) # 当天感染者人数
    R = N if R+dR*dt>=N else (0 if R+dR*dt<=0 else R+dR*dt) # 当天康复者人数
    return [S, I, R]

def calculate(func,sir,days):
    dt = 1
    t = np.arange(0,days,dt) # 设置时间步
    res=[]
    for itm in t:
        sir=func(sir,dt)      # 运行SI模型函数
        res.append(sir)      # 存储每天人数结果
    return np.array(res)

# 画图函数
def plot_graph(np_res):
    plt.figure(figsize=(10,6),dpi=300)
    plt.plot(np_res[:,0])
    plt.plot(np_res[:,1])
    plt.plot(np_res[:,2])
    plt.title("SIR模型")
    plt.xlabel("天数")
    plt.ylabel("人数")
    plt.legend(['易感者','感染者','康复者'])
    plt.text(1,4500,'N=%d \nI=%d \nr=%2.1f \nu=%2.1f' % (N,I,r,u), bbox={'facecolor': 'red', 'alpha': 0.4, 'pad': 8})
    final = [round(x) for x in result[-1]]  # 取整表示
    plt.text(45,final[0],final[0])
    plt.text(46,final[1],final[1])
    plt.text(45,final[2],final[2])
    plt.show()

# 赋初值、绘图
N = 10000
I = 1
r = 1
u = 0.8
days = 100

sir= [N-I,    # 易感人数
       I,     # 感染人数
       0]     # 康复人数

result = calculate(SIR,sir,days)
plot_graph(result)
final = [round(x) for x in result[-1]]  # 取整表示
print(final)
