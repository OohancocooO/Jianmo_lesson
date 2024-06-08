from scipy.optimize import linprog
c=[-72, -64]    #目标向量
A =[[1, 1],[12, 8]]
b1=[[50],[480]]
bound=((0,100/3.0),(0,None))
res1=linprog(c,A,b1,None,None,bound,method='simplex')
b2=[[50],[480+1]]
bound=((0,100/3.0),(0,None))
res2=linprog(c,A,b2,None,None,bound,method='simplex')
print(res1.fun-res2.fun)
