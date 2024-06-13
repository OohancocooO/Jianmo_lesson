%% �Ŵ��㷨��⺯����Сֵ
clc, clear

% �����Ŵ��㷨ѡ��
options = gaoptimset;
options.PopulationType = 'doubleVector';
options.PopInitRange = [0; 2 * pi];
options.PopulationSize = 300;
options.StallGenLimit = inf;
options.StallTimeLimit = inf;
options.PlotFcns = @gaplotbestf;
options.Generations = 100;

% ʹ���Ŵ��㷨�����Сֵ��
[x, fval, reason] = ga(@f, 1, options);

fprintf('������ [0, 2��] �ϣ���������Сֵ���� x = %.4f����Ӧ����Сֵ�� f(x) = %.4f\n', x, fval);
fprintf('ֹͣ��ԭ���ǣ�%s\n', reason);