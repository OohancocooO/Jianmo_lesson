import numpy as np  # 科学计算工具包
import matplotlib.pyplot as plt  # 画图工具包

# 设置matplotlib支持中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

def SEIJR(seijr,dt):
    S,E,I,J,R = seijr
    dS = -(r*E*S)/N                  # 易感者微分方程
    dE = (r*E*S)/N - a*E -p*E        # 潜伏者微分方程
    dI = a*E - u*I                   # 感染者微分方程
    dJ = t*I                         # 确诊者微分方程
    dR = p*E + u*I                   # 康复者微分方程
    S = 0 if S+dS*dt<=0 else S+dS*dt                        # 当天易感者人数
    E = 0 if E+dE*dt<=0 else (N if E+dE*dt>=N else E+dE*dt) # 当天潜伏者人数
    I = N if I+dI*dt>=N else (0 if I+dI*dt<=0 else I+dI*dt) # 当天感染者人数
    J = N if J+dJ*dt>=N else (0 if J+dJ*dt<=0 else J+dJ*dt) # 当天感染者人数
    R = N if R+dR*dt>=N else (0 if R+dR*dt<=0 else R+dR*dt) # 当天康复者人数
    return [S, E, I, J, R]

def calculate(func,seijr,days):
    dt = 1
    t = np.arange(0,days,dt) # 设置时间步
    res=[]
    for itm in t:
        seijr=func(seijr,dt)      # 运行SEIR模型函数
        res.append(seijr)       # 存储每天人数结果
    return np.array(res)

# 画图函数
def plot_graph(np_res):
    plt.figure(figsize=(10,6),dpi=300)
#     plt.plot(np_res[:,0])
    plt.plot(np_res[:,1])
    plt.plot(np_res[:,2])
    plt.plot(np_res[:,3])
    plt.plot(np_res[:,4])
    plt.title("SEIJR模型")
    plt.xlabel("天数")
    plt.ylabel("人数")
    plt.legend(['潜伏者','感染者','确诊者','康复者']) # '易感者',
    plt.text(1,500,'N=%d \nE=%d \nr=%3.2f \na=%3.2f \nu=%3.2f' % (N,E,r,a,u), bbox={'facecolor': 'red', 'alpha': 0.4, 'pad': 8})
    final = [round(x) for x in result[-1]]  # 取整表示
#     plt.text(90,final[0]+100,'%d' % final[0])  # 易感人数
#     plt.text(90,final[1]+100,'%d' % final[1])  # 潜伏人数
    plt.text(days-10,final[2]-50,'%d' % final[2])  # 感染人数
    plt.text(days-10,final[3]-50,'%d' % final[3])  # 确诊人数
#     plt.text(90,final[4]+100,'%d' % final[4])  # 康复人数
    plt.show()

# 赋初值、绘图
N = 14000000
E = 1000
I = 100
J = 41
r = 0.25
a = 0.2
u = 0.1
p = 0.08
t = 0.2
days = 200

seijr = [N-E,    # 易感人数
         E,      # 潜伏人数
         I,      # 感染人数
         J,      # 确诊人数
         0]      # 康复人数

result = calculate(SEIJR,seijr,days)
plot_graph(result)
final = [round(x) for x in result[-1]]  # 取整表示
print(final)
