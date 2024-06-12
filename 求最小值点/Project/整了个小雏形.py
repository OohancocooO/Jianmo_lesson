#可以自我尝试100次，自己搜索最适合最小值的种子数后，再带入数值得解，画出位置
#就是运行比较慢
import numpy as np
import matplotlib.pyplot as plt

# 定义目标函数
def f(x):
    return 11 * np.sin(6 * x) + 7 * np.cos(5 * x)

# 遗传算法参数设置
pop_size = 100        # 种群大小
generations = 1000    # 进化代数
mutation_rate = 0.01  # 变异率
crossover_rate = 0.7  # 交叉率
domain = (0, 2 * np.pi)  # 定义搜索空间

# 遗传算法函数
def genetic_algorithm(seed):
    np.random.seed(seed)
    # 初始化种群
    population = np.random.uniform(domain[0], domain[1], pop_size)

    # 适应度函数：将目标函数值取反，并转化为非负值
    def fitness(x):
        return -f(x) + 20  # 加一个足够大的常数使适应度值为非负

    # 选择操作
    def selection(population, fitness_vals):
        probabilities = fitness_vals / fitness_vals.sum()
        indices = np.random.choice(np.arange(pop_size), size=pop_size, p=probabilities)
        return population[indices]

    # 交叉操作
    def crossover(parent1, parent2):
        if np.random.rand() < crossover_rate:
            point = np.random.randint(1)
            return parent1 + point * (parent2 - parent1)
        else:
            return parent1

    # 变异操作
    def mutate(child):
        if np.random.rand() < mutation_rate:
            child = np.random.uniform(domain[0], domain[1])
        return child

    # 遗传算法主循环
    for generation in range(generations):
        # 计算适应度
        fitness_vals = fitness(population)
        
        # 选择
        selected_population = selection(population, fitness_vals)
        
        # 生成下一代
        new_population = []
        for i in range(0, pop_size, 2):
            parent1, parent2 = selected_population[i], selected_population[i+1]
            child1, child2 = crossover(parent1, parent2), crossover(parent2, parent1)
            new_population.extend([mutate(child1), mutate(child2)])
        
        population = np.array(new_population)
        
    # 输出最终结果
    best_individual = population[np.argmin(fitness(population))]
    best_value = f(best_individual)
    return best_value, best_individual

# 尝试多个种子并记录最佳结果
best_value_overall = float('inf')
best_individual_overall = None
best_seed = None

for seed in range(100):
    best_value, best_individual = genetic_algorithm(seed)
    if best_value < best_value_overall:
        best_value_overall = best_value
        best_individual_overall = best_individual
        best_seed = seed

print(f"Best Seed = {best_seed}")
print(f"Best Value = {best_value_overall}, Best Individual = {best_individual_overall}")

# 画出函数图像和最小值点
x = np.linspace(0, 2 * np.pi, 1000)
y = f(x)

plt.plot(x, y, label='f(x) = 11sin(6x) + 7cos(5x)')
plt.scatter(best_individual_overall, best_value_overall, color='red', label=f'Minimum point ({best_individual_overall:.4f}, {best_value_overall:.4f})')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.title('Genetic Algorithm Optimization')
plt.show()
