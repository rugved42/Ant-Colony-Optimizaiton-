#!/usr/bin/env python
import numpy as np
from numpy import inf
import matplotlib.pyplot as plt
locations = []
f = open("berlin52.txt", "r+")
u = f
for line in u:
    line = line.split(" ", 1)[1]
    y = line.split()
    y[0] = float(y[0])
    y[1] = float(y[1])
    locations.append(y)
print(locations)

dist = np.zeros((52,52))
for i,city in enumerate(locations):
    for j,city2 in enumerate(locations): 
        dist[i][j] = np.linalg.norm(np.array(city)-np.array(city2))

d = dist
iteration = 1000
n_city = 52 
n_ants = 52
total_p = 1
pheromone_per_path = total_p/d
pheromone_per_path[pheromone_per_path == inf] = 0
print(pheromone_per_path)
alpha = 0.85 
beta = 1.7
ini_p = np.ones((n_ants, n_city))
rute = np.ones((n_ants, n_city+1))
run = 0
bestp = []
while run < 1:
    m = n_ants
    n = n_city
    pheromne = ini_p
    e = .2         
    alpha = 1    
    beta = 1.8 
    visibility = pheromone_per_path
    for ite in range(iteration):
        rute[:,0] = 1         
        for i in range(m):
            temp_visibility = np.array(visibility)        
            for j in range(n-1):
                #print(rute)
                combine_feature = np.zeros(52)     
                cum_prob = np.zeros(52)            
                cur_loc = int(rute[i,j]-1)        
    #             print("cur_loc",cur_loc)
                temp_visibility[:,cur_loc] = 0     
    #             print(temp_visibility)
    #             print("pheromne",pheromne[cur_loc,:])
                p_feature = np.power(pheromne[cur_loc,:],beta)        
                v_feature = np.power(temp_visibility[cur_loc,:],alpha)  
                p_feature = p_feature[:,np.newaxis]                     
                v_feature = v_feature[:,np.newaxis]                     
                combine_feature = np.multiply(p_feature,v_feature)     
                total = np.sum(combine_feature)                        
                probs = combine_feature/total   
                cum_prob = np.cumsum(probs)    
                #print(cum_prob)
                r = np.random.random_sample()   
                #print(r)
                city = np.nonzero(cum_prob>r)[0][0]+1       
                #print(city)
                rute[i,j+1] = city              
            left = list(set([i for i in range(1,n+1)])-set(rute[i,:-2]))[0]   
            rute[i,-2] = left                   
        rute_opt = np.array(rute)              
        dist_cost = np.zeros((m,1)) 
        for i in range(m):
            s = 0
            for j in range(n-1):
                s = s + d[int(rute_opt[i,j])-1,int(rute_opt[i,j+1])-1]  
            dist_cost[i]=s                      
        dist_min_loc = np.argmin(dist_cost)             
        dist_min_cost = dist_cost[dist_min_loc]        
        best_route = rute[dist_min_loc,:]               
        pheromne = (1-e)*pheromne                      
        for i in range(m):
            for j in range(n-1):
                dt = 1/dist_cost[i]
                pheromne[int(rute_opt[i,j])-1,int(rute_opt[i,j+1])-1] = pheromne[int(rute_opt[i,j])-1,int(rute_opt[i,j+1])-1] + dt   
        best_dist = int(dist_min_cost[0]) + d[int(best_route[-2])-1,0]
        bestp.append(best_dist)
    print('route of all the ants at the end :')
    print(rute_opt)
    print()
    print('best path :',best_route)
    print('cost of the best path',int(dist_min_cost[0]) + d[int(best_route[-2])-1,0])
    plt.plot(bestp)
    f = open("berlin_52_E.txt", "a")
    f.write("Run {}/{} Length of best path: {}\n".format(run, 10,int(dist_min_cost[0]) + d[int(best_route[-2])-1,0]))
    run +=1

plot_path = []
for i in best_route: 
    plot_path.append(list(locations[int(i) - 1])) 
print(plot_path)
num = np.arange(1,53)
x,y = [],[]
for start, end in plot_path:
    x.append(start)
    y.append(end)
print(x,y)

fig, ax = plt.subplots()
ax.plot(x, y,'-o')
for i, num in enumerate(num):
    ax.annotate(num, (x[i], y[i]))

t = open("aco_opti.txt", "r+")
opti = []
for lines in t:
    lines.split()
    opti.append(lines.strip())
opti[-1] = '1'
opti = [int(i) for i in opti]
# print(opti)
opti_path,xo,yo = [],[],[]
for i in opti: 
    opti_path.append(locations[int(i) - 1])
    xo.append(locations[int(i) - 1][0])
    yo.append(locations[int(i) - 1][1])
fig, ax = plt.subplots()
ax.plot(xo, yo, '-o')
for i, num in enumerate(num):
    ax.annotate(num, (xo[i], yo[i]))

