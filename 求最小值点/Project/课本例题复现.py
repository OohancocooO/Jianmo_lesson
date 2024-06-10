import numpy as np
import matplotlib.pyplot as plt
from deap import algorithms, base, creator, tools

# 设置matplotlib支持中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
# 参数设置
opt_minmax = 1  # 目标优化类型：1最大化 -1最小化
num_ppu = 50  # 种群规模
num_gen = 60  # 最大遗传代数
len_ch = 1  # 基因长度
gap = 0.9  # 代沟
sub = -1  # 变量取值下限
up = 2.5  # 变量取值上限
trace = np.zeros((num_gen, 2))  # 遗传迭代性能跟踪器


# 目标函数
def fun_sigv(x):
    return x * np.sin(10 * np.pi * x) + 2


# 设置遗传算法工具箱
creator.create("FitnessMax", base.Fitness, weights=(opt_minmax,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, sub, up)
toolbox.register(
    "individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len_ch
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxBlend, alpha=0.5)  # 使用适合基因数为1的交叉操作
toolbox.register(
    "mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2
)  # 使用Gaussian变异
toolbox.register("select", tools.selRoulette)
toolbox.register("evaluate", fun_sigv)


# 遗传算法主程序
def main():
    pop = toolbox.population(n=num_ppu)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("mean", np.mean)

    for gen in range(num_gen):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=gap, mutpb=0.1)
        fits = list(map(toolbox.evaluate, [ind[0] for ind in offspring]))

        for fit, ind in zip(fits, offspring):
            ind.fitness.values = (fit,)

        hof.update(offspring)
        record = stats.compile(offspring)
        trace[gen, 0] = record["max"]
        trace[gen, 1] = record["mean"]

        pop[:] = toolbox.select(offspring, k=len(pop))

    tx = np.arange(sub, up, 0.01)
    plt.plot(tx, fun_sigv(tx))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("一元函数优化结果")
    plt.plot(hof[0][0], hof[0].fitness.values[0], "r*")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(trace[:, 0], "r-*", label="各子代种群最优解")
    ax.plot(trace[:, 1], "b-o", label="各子代种群平均值")
    ax.legend()
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("目标函数优化情况")
    ax.set_title("一元函数优化过程")
    plt.show()


if __name__ == "__main__":
    main()
