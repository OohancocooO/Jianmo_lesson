% 定义目标函数
objFunc = @f;

% 定义变量的上下界
lb = 0; % 下界
ub = 2 * pi; % 上界

% 使用遗传算法求解最小值点
% 'ga' 函数的第一个参数是目标函数
% 第二个参数是变量的数量，这里是1
% 'ga' 函数的第三、第四个参数分别是下界和上界
options = optimoptions('ga', 'Display', 'iter');
[x_min, fval] = ga(objFunc, 1, [], [], [], [], lb, ub, [], options);

% 显示结果
fprintf('在区间 [0, 2π] 上，函数的最小值点是 x = %.4f，对应的最小值是 f(x) = %.4f\n', x_min, fval);
