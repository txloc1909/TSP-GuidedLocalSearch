import numpy as np 
import random
import math

def readData (path):
    datafile = open(path, 'r')

    Name = datafile.readline().strip().split()[-1]

    datafile.readline()
    datafile.readline()
    datafile.readline()

    Type = datafile.readline().strip().split()[-1]
    Dimension = datafile.readline().strip().split()[-1]
    Weight_Type = datafile.readline().strip().split() [-1]
    datafile.readline()

    N = int(Dimension)
    nodelist = []
    for i in range(N):
        x, y = datafile.readline().strip().split()[1:]
        nodelist.append([float(x), float(y)])
    
    return np.array(nodelist, dtype = np.float64), Name


# Calculate the distance matrix
def WeightMatrix (coord: np.ndarray):
    N = coord.shape[0]
    matrix = []

    for i in range(N):
        row = []
        for j in range(N):
            x = math.sqrt((coord[i][0] - coord[j][0])**2 + (coord[i][1] - coord[j][1])**2)
            row.append(x)
        matrix.append(row)
    
    return np.array(matrix, dtype = np.float64)


def TourLength (weight: np.ndarray, solution: list):
    length = weight[solution[-1], 0] + weight[0, solution[0]]

    for i in range(len(solution) - 1):
        length += weight[solution[i], solution[i+1]]

    return length


def initSolution (n_cities: int):
    s = np.arange(1, n_cities)
    random.shuffle(s)
    return list(s)


# Features: whether the solution has the path from 0 to i, i = 1,2,...n_cities-1
def hasFeature(solution, nextcity): # indicate whether the solution exhibit the feature
    return True if solution[0] == nextcity or solution[-1] == nextcity else False


def LocalSearch (weight: np.ndarray, init_s: list, penalty: np.ndarray, eta: float, max_unimproved: int):
    n_cities = weight.shape[0]
    s = init_s
    n_unimproved = 0

    s.append(0)
    
    while n_unimproved < max_unimproved:
        # pick city to swap
        while True:
            a, c = random.choices(range(len(s)), k = 2)
            if c - a > 1:
                break
        
        # evaluate changes in tour length and penalty
        if s[c] == s[-1]:
            before = weight[s[a], s[a+1]] + weight[s[c], s[0]]
            after = weight[s[a], s[c]] + weight[s[a+1], s[0]]
            dpen = penalty[s[a]-1] - penalty[s[0]-1]
        elif s[c+1] == s[-1]:
            before = weight[s[a], s[a+1]] + weight[s[c], s[c+1]]
            after = weight[s[a], s[c]] + weight[s[a+1], s[c+1]]
            dpen = penalty[s[a+1]-1] - penalty[s[c]-1]
        else:
            before = weight[s[a], s[a+1]] + weight[s[c], s[c+1]]
            after = weight[s[a], s[c]] + weight[s[a+1], s[c+1]]
            dpen = 0

        # decide whether to swap
        if after + eta * dpen < before:
            # swap
            head_s = s[:a+1]
            mid_s = s[a+1: c+1]
            mid_s.reverse()
            tail_s = s[c+1:]
            s = head_s + mid_s + tail_s

            #modify 0 to the last
            while s[-1] != 0:
                tmp = s[0]
                for i in range(len(s)-1):
                    s[i] = s[i+1]
                s[-1] = tmp
            
            n_unimproved = 0
        else:
            n_unimproved += 1
        
        #print(TourLength(weight, s))
    
    s.remove(0)
    return s


def GuidedLocalSearch (weight: np.ndarray, eta: float, max_iter = 10000, max_unimproved = 100):
    n_cities = weight.shape[0]

    penalty = np.zeros(n_cities-1)
    utility = np.zeros(n_cities-1)

    n_iter = 0
    n_unimproved = 0

    current_s = initSolution(n_cities)
    new_s = LocalSearch(weight, init_s = current_s, penalty = penalty, eta = eta, max_unimproved = max_unimproved)

    while n_iter < max_iter or n_unimproved < max_unimproved:
        
        if TourLength(weight, new_s) < TourLength(weight, current_s):
            current_s = new_s
            n_unimproved = 0
        else:
            n_unimproved += 1
        

        for i in range(len(utility)):
            if hasFeature(current_s, i+1):
                utility[i] = weight[0, i+1] / (1 + penalty[i])
        
        penalty = np.where(penalty == max(penalty), penalty+1, penalty)
        
        new_s = LocalSearch(weight, init_s= current_s, penalty = penalty, eta = eta, max_unimproved = max_unimproved)
        
        n_iter += 1

        #if n_iter % 100 == 0:
            #print("Current: %.2f New: %.2f" %(TourLength(weight, current_s), TourLength(weight, new_s)))

    return new_s


# Config
eta = 1000
max_iter = 1000
max_unimproved = 500

path_1 = '~/metaheuristics_20191/TSP/smallset/bcl380.txt'
path_2 = '~/metaheuristics_20191/TSP/smallset/pbk411.txt'
path_3 = '~/metaheuristics_20191/TSP/smallset/pbl395.txt'
path_4 = '~/metaheuristics_20191/TSP/smallset/pbm436.txt'
path_5 = '~/metaheuristics_20191/TSP/smallset/pbn423.txt'

paths = [path_1, path_2, path_3, path_4, path_5]

csvData = [['Data set', 'Best', 'Worst', 'Average', 'Deviation']]

for path in paths:
    coord, name = readData(path)
    w = WeightMatrix(coord)

    history = np.empty(30)
    for i in range(30):
        opt = GuidedLocalSearch(w, eta, max_iter, max_unimproved)
        history[i] = TourLength(w, opt)
    
    best = min(history)
    worst = max(history)
    avg = np.mean(history)
    deviation = math.sqrt(np.mean((history - avg)**2))
    row = [name, int(best), int(worst), int(avg), int(deviation)]
    csvData.append(row)

import csv
with open("tsp_smallset.csv", "w") as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csvData)

csvFile.close()