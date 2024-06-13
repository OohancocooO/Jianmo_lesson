%% 遗传算法求解函数最小值
clc, clear

% 定义遗传算法选项
options = gaoptimset;
options.PopulationType = 'doubleVector';
options.PopInitRange = [0; 2 * pi];
options.PopulationSize = 300;
options.StallGenLimit = inf;
options.StallTimeLimit = inf;
options.PlotFcns = @gaplotbestf;
options.Generations = 100;

% 使用遗传算法求解最小值点
[x, fval, reason] = ga(@f, 1, options);

fprintf('在区间 [0, 2π] 上，函数的最小值点是 x = %.4f，对应的最小值是 f(x) = %.4f\n', x, fval);
fprintf('停止的原因是：%s\n', reason);