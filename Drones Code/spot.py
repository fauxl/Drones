import matplotlib.pyplot as plt
import random

import math


class spot:
    def __init__(self, i, j):
        self.set = 0 # For coloring purpose on the graph
        self.i = i          
        self.j = j
        self.f = 0      # f is the cost function f = g+h
        self.g = 0      # g is like the actual distance from the starting point to the node we are
        self.h = 0      # h is the heuristic - estimates the distance from the node we are to the end for each possibile node - here it has been used euclidean distance
        self.bat = 1
        self.tim = 1
        self.value = 1
        self.utility=0
        self.weiba = 0
        self.weita = 0
        self.important = 1
        self.signeed = random.uniform(0.0,1.0)
        self.intervalue = 0

        self.parent = None          # To trace back the path, we need to record parents
        self.wall = False           # spot obstacle set false
        self.isdrone = False
        self.signal = False
        self.path=False
        self.crow = False
        self.recharge = False
        self.interf = False
        self.altitud = False

        self.neighbors = []         # list of neighbors of a spot              
       
        if random.random() < 0.005:   # Percentage of recharge zone
            if self.value==1:
                self.recharge = True
                self.value=0.9

        if random.random() < 0.1:   # Percentage of minimum altitude required
            if self.value==1:
                self.altitud= True
                self.value=0.7

        if random.random() < 0.2:   # Percentage of obstacles or not linked spots generated randomly
            if self.value==1:
                self.wall = True
                self.value=0

        if random.random() < 0.1:   # Percentage of crowded places
            if self.value==1:
                self.crow = True
                self.value=0.5

    # Neighbor spot adding
    def add_neighbors(self, grid_passed,size):
        i = self.i
        j = self.j
        if j < size - 1:
            self.neighbors.append(grid_passed[i][j + 1])
        if i > 0:
            self.neighbors.append(grid_passed[i - 1][j])
        if j > 0:
            self.neighbors.append(grid_passed[i][j - 1])
        if i < size - 1:
            self.neighbors.append(grid_passed[i + 1][j])
        if i < size - 1 and j < size - 1:
            self.neighbors.append(grid_passed[i + 1][j + 1])
        if i > 0 and j < size - 1:
            self.neighbors.append(grid_passed[i - 1][j + 1])
        if i < size - 1 and j > 0:
            self.neighbors.append(grid_passed[i + 1][j - 1])
        if i > 0 and j > 0:
            self.neighbors.append(grid_passed[i - 1][j - 1])
            # Heuristic Euclidean Distance

    def heuristic(self, b):                                   
        dist = math.sqrt((self.i - b.i)**2 + (self.j - b.j)**2)
        return dist

