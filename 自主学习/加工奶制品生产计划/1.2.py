from scipy.optimize import linprog
c=[-72, -64, 35]    #目标向量
A =[[1, 1, -1],[12, 8, 0]]; b=[[50],[480]]
bound=((0,100/3.0),(0,None),(0,None))
res=linprog(c,A,b,None,None,bound,method='simplex')
print('最优值:',res.fun)
print('最优解:',res.x)
