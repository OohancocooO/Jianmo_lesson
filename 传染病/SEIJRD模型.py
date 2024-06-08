import numpy as np  # 科学计算工具包
import matplotlib.pyplot as plt  # 画图工具包

# 设置matplotlib支持中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

def SEIJRD(seijrd, para, dt):
    S, E, I, J, R, D = seijrd
    r, a, u, p, t, w = para  # 赋值
    dS = -(r * E * S) / N  # 易感者微分方程
    dE = (r * E * S) / N - a * E - p * E  # 潜伏者微分方程
    dI = a * E - u * I - w * I  # 感染者微分方程
    dJ = t * I  # 确诊者微分方程
    dR = p * E + u * I  # 康复者微分方程
    dD = w * I  # 死亡者微分方程

    S = 0 if S + dS * dt <= 0 else S + dS * dt  # 当天易感者人数
    E = 0 if E + dE * dt <= 0 else (N if E + dE * dt >= N else E + dE * dt)  # 当天潜伏者人数
    I = N if I + dI * dt >= N else (0 if I + dI * dt <= 0 else I + dI * dt)  # 当天感染者人数
    J = N if J + dJ * dt >= N else (0 if J + dJ * dt <= 0 else J + dJ * dt)  # 当天感染者人数
    R = N if R + dR * dt >= N else (0 if R + dR * dt <= 0 else R + dR * dt)  # 当天康复者人数
    D = N if D + dD * dt >= N else (0 if D + dD * dt <= 0 else D + dD * dt)  # 当天死亡者人数
    return [S, E, I, J, R, D]


def calculate(func, seijrd, days):
    dt = 1
    t = np.arange(0, days, dt)  # 设置时间步
    res = []
    for itm in t:
        if itm < 13:
            seijrd = func(seijrd, para1, dt)  # 运行SEIJR模型函数
        else:
            seijrd = func(seijrd, para2, dt)  # 运行SEIJR模型函数
        res.append(seijrd)  # 存储每天人数结果
    return np.array(res)


# 画图函数
def plot_graph(np_res):
    plt.figure(figsize=(10, 6), dpi=300)
    ax = plt.subplot(111)
    #     plt.plot(np_res[:,0])
    plt.plot(np_res[:, 1])
    plt.plot(np_res[:, 2])
    plt.plot(np_res[:, 3])
    plt.plot(np_res[:, 4])
    plt.plot(np_res[:, 5])
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels(['0110', '0130', '0209', '0229', '0320', '0409'])
    plt.title("武汉市COVID-19疫情预测")
    plt.xlabel('日期')
    plt.ylabel("人数")
    plt.legend(['潜伏者', '感染者', '确诊者', '康复者', '死亡者'])  # '易感者',
    plt.text(60, 20000, 'N=%d, r1=%3.2f, r2=%3.2f' \
             % (14000000, 0.52, 0.2), \
             bbox={'facecolor': 'red', 'alpha': 0.4, 'pad': 8})
    plt.text(60, 15000, r'$\alpha$=%3.2f, $\mu$=%3.2f, $\pi$=%3.2f, $\tau$=%3.2f, $\omega$=%3.2f' \
             % (0.2, 0.1, 0.1, 0.18, 0.009), \
             bbox={'facecolor': 'red', 'alpha': 0.4, 'pad': 8})
    final = [round(x) for x in result[-1]]  # 取整表示
    #     plt.text(90,final[0]+100,'%d' % final[0])  # 易感人数
    #     plt.text(90,final[1]+100,'%d' % final[1])  # 潜伏人数
    plt.text(days - 20, final[2] + 100, '感染人数为 %d' % final[2])  # 感染人数
    plt.text(days - 20, final[3] + 100, '确诊人数 %d' % final[3])  # 确诊人数
    plt.text(days - 20, final[4] + 100, '康复人数为 %d' % final[4])  # 康复人数
    plt.text(days - 20, final[5] + 100, '死亡人数为 %d' % final[5])  # 死亡人数

    plt.show()


# 武汉
N = 14000000
E = 800
I = 100
J = 41
days = 100
para1 = [0.52, 0.2, 0.1, 0.1, 0.18, 0.009]
para2 = [0.2, 0.2, 0.1, 0.1, 0.18, 0.009]

seijrd = [N - E,  # 易感人数
          E,  # 潜伏人数
          I,  # 感染人数
          J,  # 确诊人数
          0,  # 死亡人数
          0]  # 康复人数

result = calculate(SEIJRD, seijrd, days)
plot_graph(result)
final = [round(x) for x in result[-1]]  # 取整表示
print(final)
