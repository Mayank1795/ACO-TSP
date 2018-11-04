import numpy as np
import math
import networkx as nx
import copy
#import matplotlib.pyplot as plt

# no. of cities = no. of agents
n = 5
ant = [ i for i in range(0, n)]

# nxn distance matrix
#E = np.random.randint(low=10, high=101, size=(n,n))
E = [[0,0,0,0,0],
    [3,0,0,0,0 ],
    [6,5,0,0,0],
    [2,2,6,0,0],
    [3,3,4,6,0]]

E = np.array(E)

np.fill_diagonal(E, 0)

D = np.tril(E) + np.tril(E).T


g = np.ones((n, n))
np.fill_diagonal(g, 0)

#visualize
G = nx.from_numpy_array(g)
#nx.draw(G, with_labels=True)
#plt.show()

t = 0       # time
nc = 0      # no. of cycles
c = 1       # intial pheromone choose randomly
s = 0       # index position in track_visits rows
Q = 1       # some constant
rho =  0.5  # % of pheromone remains after decay
alpha = 0.7    # control pheromone
beta = 0.7    # control visibility
current_tc = [0]*n   # Lk for all ants
shortest_tour = np.ones((1,n), dtype=int)
shortest_tc = 0         # shortest tour cost
delta_tau = np.zeros((n,n,n), dtype=float)

delta_tau_init = np.zeros((n,n), dtype=float)

track_visit = np.zeros((n,1), dtype=int)   # each row for one ant
tau = np.tril(np.ones((n,n), dtype=float)) + np.tril(np.ones((n, n), dtype=float)).T    # pheromone on edges
current_visit = [ i for i in ant]
visibility = np.empty((n, n), dtype=float)

for i in range(n):
    for j in range(n):
        if(i!=j):
            visibility[i,j] = 1/D[i,j]

np.fill_diagonal(visibility, 0)
np.fill_diagonal(tau, 0)

print('Distance: \n', D)
print('Visibility: \n', visibility)

def initialize(n, c):
    """
        Place each ant on one of the nodes. Set intial pheromone values. 
    """   
    global tau, track_visit, s
    tau = c*tau
    print('Tau: \n', tau)
    for i in ant:
        track_visit[i,0] = i
    return

initialize(n, c)

p,q = track_visit.shape

def findNextMove(a):
    global current_visit
    # allowed towns for ant a.
    record = list(track_visit[a,:])
    visited = []
    
    for i in range(q):
        visited.append(record[i])
    
    allowed = []
    for i in range(n):
        if(i not in visited):
            allowed.append(i)
    
    #print('Ant: ', a,' Not visited: ',allowed)
    i = visited[-1]

    denom = 0
    for j in allowed:
        denom+= math.pow(tau[i, j] , alpha) * math.pow(visibility[i, j] , beta)
    
    # dict key=j, value=probability
    city_prob = {}
    for j in allowed:
        num = math.pow(tau[i, j] , alpha) * math.pow(visibility[i, j] , beta)
        e = num/denom
        city_prob[j] = e
       
    #print('City: ', max(city_prob, key = city_prob.get))
    city = max(city_prob, key = city_prob.get)
    return city

def updateLk(tour, k):
    global current_tc
    total_cost = 0
    
    for i in range(n-1):
        total_cost += D[tour[i], tour[i+1]]

    total_cost+=D[tour[-1], tour[0]]
    current_tc[k] = total_cost
    return total_cost
        
def edgeInVisited(ct, i, j):
    path = track_visit[ct,:]

    # find i and j positions, difference between their index should be 1, -1
    loc_i = np.where(path == i)     # tuple
    loc_j = np.where(path == j)
    
    if(loc_i[0] == 0 and loc_j[0] == (n-1)):
        return True
    elif(loc_i[0] == (n-1) and loc_j[0] == 0):
        return True
    elif(abs(loc_i[0] - loc_j[0]) == 1):
        return True
    else:
        return False


while(nc < 5):

    print('Iteration no.: ', nc+1)
    while(q!=5):
        s+=1
        new_cities = np.zeros((n,1), dtype=int)

        print(current_visit)

        for k in ant:
            # choose the next move // update current_visit
            chosen_city = findNextMove(k)
            #print('Selected city: ',chosen_city)
            # move the ant to next town
            current_visit[k] = chosen_city
            new_cities[k, 0] = chosen_city
        
        # update tract_visit
        new_t = np.append(track_visit, new_cities, axis=1)
        track_visit = copy.deepcopy(new_t)
        q+=1
        
    print(track_visit)

    tp = []
    for k in ant:
        tc = updateLk(track_visit[k,:], k)
        tp.append(tc)
        if((len(tp) >= 1) and (tc <= min(tp))):         # takes the latest tour if there is a tie
            shortest_tc = tc
            shortest_tour = track_visit[k,:]

        # update tau for each edge
    print(current_tc)
    print(shortest_tour)
    
    for k in range(n):
        '''if(k == n-1):
            j = 0
        else:
            j = k+1'''
        for j in range(n):
            if(k!=j):
                for a in ant:
                    if(edgeInVisited(a, k, j)):
                        delta_tau[k, j, a] = Q/(current_tc[a])
                    else:
                        delta_tau[k, j, a] = 0

                    delta_tau_init[k, j]+=delta_tau[k, j, a]
    
    for k in range(n):
        for j in range(n):
            if(k!=j):
                tau[k, j] = (1-rho)*tau[k, j] + delta_tau_init[k, j]

    print(tau)
    #nc+=1
    nc = 5



