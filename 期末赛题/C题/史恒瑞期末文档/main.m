% 读取定价数据
B = xlsread("C:\Users\86184\Desktop\S1.xlsx", 'sheet1', 'A2:F2');
disp('B size:'); disp(size(B));

% 读取未来一周的销量预测数据和平均损耗率数据
S = xlsread("C:\Users\86184\Desktop\未来一周预测.xlsx", 'sheet1', 'B2:G8');
L = xlsread("C:\Users\86184\Desktop\平均损耗率.xlsx", 'sheet1', 'B2:B7');

%% 参数初始化
narvs = 2; % 变量个数
T0 = 100; % 初始温度
T = T0; % 迭代中温度会发生改变，第一次迭代时温度就是T0
maxgen = 200; % 最大迭代次数

Lk = 100; % 每个温度下的迭代次数
alfa = 0.95; % 温度衰减系数
x_lb = [-3 0]; % x的下界
x_ub = [3 7]; % x的上界

%% 随机生成一个初始解矩阵
x0 = zeros(1, narvs);
for i = 1:narvs
    x0(i) = x_lb(i) + (x_ub(i) - x_lb(i)) * rand(1);
end
y0 = Obj_fun2(x0, B, S, L); % 计算当前解的函数值

%% 定义一些保存中间过程的量，方便输出结果和画图
max_y = y0;
best_x = x0; % 初始化best_x
MAxY = zeros(maxgen, 1); % 记录每一次外层循环结束后找到的max_y（方便画图）

%% 模拟退火过程
for iter = 1:maxgen % 外循环，我这里采用的是指定最大迭代次数，用于温度降低
    for i = 1:Lk % 内循环，在每个温度下开始迭代，用于广泛搜索新解
        y = randn(1, narvs); % 生成1行narvs列的N(0,1)随机数
        z = y / sqrt(sum(y.^2)); % 根据新解的产生规则计算z
        x_new = x0 + z * T; % 根据新解的产生规则计算新解x_new的值
        % 判断新解是否在定义域内，如果这个新解的位罩超出了定义域，就对其进行调整
        for j = 1:narvs
            if x_new(j) < x_lb(j)
                r = rand(1);
                x_new(j) = r * x_lb(j) + (1 - r) * x0(j);
            elseif x_new(j) > x_ub(j)
                r = rand(1);
                x_new(j) = r * x_ub(j) + (1 - r) * x0(j);
            end
        end
        x1 = x_new;
        % 将调整后的x_new赋值给新解x1
        y1 = Obj_fun2(x1, B, S, L); % 计算新解的函数值
        if y1 > y0
            % 如果新解函数值大于当前解的函数值
            x0 = x1; % 更新当前解为新解
            y0 = y1;
        else
            p = exp(-(y0 - y1) / T); % 根据Metropolis准则计算一个概率
            if rand(1) < p % 生成一个随机数和这个概率比较，如果该随机数小于这个概率
                x0 = x1; % 更新当前解为新解
                y0 = y1;
            end
        end
        % 判断是否要更新找到的最佳的解
        if y0 > max_y % 如果当前解更好，则对其进行更新
            max_y = y0; % 更新最大的y
            best_x = x0; % 更新找到的最好的x
        end
    end
    MAxY(iter) = max_y; % 保存本轮外循环结束后找到的最大的y
    T = alfa * T; % 温度下降
end

disp('最佳的位置是：'); disp(best_x) % 输出的为最后多次迭代不变的解
disp('此时最优值是：'); disp(max_y) % 输出的为最后多次迭代不变的最优值

%% 目标函数定义
function w = Obj_fun2(x, B, S, L)
    % 示例目标函数，这里你可以根据具体需要来调整
    % x: 当前解
    % B: 定价数据
    % S: 未来一周的销量预测数据
    % L: 平均损耗率数据
    
    % 初始化P和R，使其与S维度一致
    [num_rows, num_cols] = size(S);
    P = ones(num_rows, num_cols); % 示例初始化，实际可以根据需要调整
    R = ones(num_rows, num_cols); % 示例初始化，实际可以根据需要调整
    
    % 初始化w
    w = 0;
    % 计算目标函数值
    for i = 1:num_rows
        for j = 1:num_cols
            % 满足约束条件
            if S(i,j) <= R(i,j) && R(i,j) <= max(S(i,:)) && ...
               1.1 * B(i) <= P(i,j) && P(i,j) <= 1.5 * B(i) && ...
               B(i) * R(i,j) * L(i) + (R(i,j) - S(i,j)) <= L(i) * max(S(i,:))
                w = w + S(i,j) * (P(i,j) - B(i)) - R(i,j) * B(i) * L(i);
            end
        end
    end
end



