function y = f( x )
%F 用于遗传算法求解函数最小值的案例
%   此处显示详细说明
if 0 < x < 2*pi
    y = 11 * sin(6 * x) + 7 * cos(5 * x);   % 有点是因为将x定义为向量（有范围）
else
    y = 0;
end

