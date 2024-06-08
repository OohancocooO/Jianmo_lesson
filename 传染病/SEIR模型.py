import numpy as np  # 科学计算工具包
import matplotlib.pyplot as plt  # 画图工具包

# 设置matplotlib支持中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# 定义SEIR函数——根据SEIR模型计算每天新增的易感人数，潜伏人数，感染人数，康复人数，返回累计人数
def SEIR(seir,dt):
    S,E,I,R = seir             # 每天初始SIR的数值
    dS = -(r*I*S)/N            # 易感者微分方程
    dE = (r*I*S)/N - a*E       # 潜伏者微分方程
    dI = a*E - u*I             # 感染者微分方程
    dR = u*I                   # 康复者微分方程
    S = 0 if S+dS*dt<=0 else S+dS*dt                        # 当天易感者人数
    E = 0 if E+dE*dt<=0 else (N if E+dE*dt>=N else E+dE*dt) # 当天潜伏者人数
    I = N if I+dI*dt>=N else (0 if I+dI*dt<=0 else I+dI*dt) # 当天感染者人数
    R = N if R+dR*dt>=N else (0 if R+dR*dt<=0 else R+dR*dt) # 当天康复者人数
    return [S, E, I, R]

def calculate(func,seir,days):
    dt = 1
    t = np.arange(0,days,dt) # 设置时间步
    res=[]
    for itm in t:
        seir=func(seir,dt)      # 运行SEIR模型函数
        res.append(seir)       # 存储每天人数结果
    return np.array(res)

# 画图函数
def plot_graph(np_res):
    plt.figure(figsize=(10,6),dpi=300)
    plt.plot(np_res[:,0])
    plt.plot(np_res[:,1])
    plt.plot(np_res[:,2])
    plt.plot(np_res[:,3])
    plt.title("SEIR模型")
    plt.xlabel("天数")
    plt.ylabel("人数")
    plt.legend(['易感者','潜伏者','感染者','康复者'])
    plt.text(1,4500,'N=%d \nE=%d  \nI=%d \nr=%2.1f \na=%2.1f \nu=%2.1f' % (N,E,I,r,a,u), bbox={'facecolor': 'red', 'alpha': 0.4, 'pad': 8})
    final = [round(x) for x in result[-1]]  # 取整表示
    plt.text(90,650,'%d' % final[0])
    plt.text(92,100,'%d' % final[2])
    plt.text(90,final[3],'%d' % final[3])
    plt.show()

# 赋初值、绘图
N = 10000
E = 0
I = 10
r = 0.6
a = 0.3
u = 0.3
days = 100

seir= [N-I-E,    # 易感人数
       E,      # 潜伏人数
       I,      # 感染人数
       0]      # 康复人数

result = calculate(SEIR,seir,days)
plot_graph(result)
final = [round(x) for x in result[-1]]  # 取整表示
print(final)
