import numpy as np

# 数据
N_fertilizer = np.array([0, 34, 67, 101, 135, 202, 259, 336, 404, 471])
N_yield = np.array(
    [15.18, 21.36, 25.72, 32.29, 34.03, 39.45, 43.15, 43.46, 40.83, 30.75]
)

P_fertilizer = np.array([0, 24, 49, 73, 98, 147, 196, 245, 294, 342])
P_yield = np.array(
    [33.46, 32.47, 36.06, 37.96, 41.04, 40.09, 41.26, 42.17, 40.36, 42.73]
)

K_fertilizer = np.array([0, 47, 93, 140, 186, 279, 372, 465, 558, 651])
K_yield = np.array(
    [18.98, 27.35, 34.86, 38.52, 38.44, 37.73, 38.43, 43.87, 42.77, 46.22]
)


def descriptive_stats(fertilizer, yield_):
    return {
        "mean_fertilizer": np.mean(fertilizer),
        "std_fertilizer": np.std(fertilizer, ddof=1),
        "min_fertilizer": np.min(fertilizer),
        "max_fertilizer": np.max(fertilizer),
        "mean_yield": np.mean(yield_),
        "std_yield": np.std(yield_, ddof=1),
        "min_yield": np.min(yield_),
        "max_yield": np.max(yield_),
    }


# 分别计算N、P、K的描述性统计分析结果
stats_N = descriptive_stats(N_fertilizer, N_yield)
stats_P = descriptive_stats(P_fertilizer, P_yield)
stats_K = descriptive_stats(K_fertilizer, K_yield)

print("N Stats:", stats_N)
print("P Stats:", stats_P)
print("K Stats:", stats_K)
