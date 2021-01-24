import matplotlib.pyplot as plt
import os
import numpy as np

from UtilityAgent import *
from spot import *

PROJECT_ROOT_DIR = "C://Users//FauxL//Documents//Drones//Drones Code//img"
CHAPTER_ID = "rl"


class GraphEnv:
    def __init__(self,size,grid):

        self.size = size
        self.grid = grid

        # Making a grid
        for i in range(size):
            row = [spot(0, 0) for i in range(size)]
            self.grid.append(row)
        
        start = grid[0][0]                  # Start can be randomized = grid[int(random.random()*40)][int(random.random()*40)]
        end = grid[size - 1][size - 1]      # Destination can be randomized = grid[int(random.random()*40)][int(random.random()*40)]
        grid[0][0].wall = False
        grid[size - 1][size - 1].wall = False       

        # Putting the spot in the grid
        for i in range(size):
            for j in range(size):
                self.grid[i][j] = spot(i, j)

        # Filling neighbours
        for i in range(size):
            for j in range(size):
                self.grid[i][j].add_neighbors(grid,size)

    def SetSignal(self,d1,beam):
        grid =  self.grid
        X = beam # R is the radius
        for x in range(-X,X+1):
            Y = int((beam*beam-(x*x))**0.5) # bound for y given x
            for y in range(d1.j-Y,d1.j+Y+1):
                grid[d1.i+x][y].interf = True
                grid[d1.i+x][y].value = 0.3
                grid[d1.i+x][y].wall = False
                grid[d1.i+x][y].recharge = False
                grid[d1.i+x][y].crow= False
                grid[d1.i+x][y].altitud= False

    def heuristic(self,a, b,c,d):                                   
        dist = math.sqrt((a - c)**2 + (b - d)**2)
        return dist

    def PutDrones(self):
        distance = 8
        size = self.size
        d1i = 22
        d1j = 28

        d2i = d1i + distance
        d2j = d1j + distance

        d3i = d1i + distance
        d3j = d1j - distance

        d4i =  d1i + distance*2
        d4j = d1j 

        drone = Drone(1,1,1,0,0,5)
        drone1 = Drone(1,1,1,d1i,d1j,6)
        drone2 = Drone(1,1,1,d2i,d2j,6)
        drone3 = Drone(1,1,1,d3i,d3j,6)

        self.grid[d1i][d1j].isdrone=True
        self.grid[d2i][d2j].isdrone=True
        self.grid[d3i][d3j].isdrone=True

        GraphEnv.SetSignal(self,drone1,drone1.beam)
        GraphEnv.SetSignal(self,drone2,drone2.beam)
        GraphEnv.SetSignal(self,drone3,drone3.beam)

        # Filling Network
        for i in range(size):
            for j in range(size):
                if(0<heuristic(self.grid[i][j],self.grid[d1i][d1j])<= drone1.beam):
                    self.grid[i][j].intervalue += 1/heuristic(self.grid[i][j],self.grid[d1i][d1j])
                    self.grid[i][j].signeed -= self.grid[i][j].intervalue
                    self.grid[i][j].value = 1-self.grid[i][j].intervalue
                if(0<heuristic(self.grid[i][j],self.grid[d2i][d2j])<= drone2.beam):
                    self.grid[i][j].intervalue += 1/heuristic(self.grid[i][j],self.grid[d2i][d2j])
                    self.grid[i][j].signeed -= self.grid[i][j].intervalue
                    self.grid[i][j].value = 1-self.grid[i][j].intervalue
                if(0<heuristic(self.grid[i][j],self.grid[d3i][d3j])<= drone3.beam):
                    self.grid[i][j].intervalue += 1/heuristic(self.grid[i][j],self.grid[d3i][d3j])   
                    self.grid[i][j].signeed -= self.grid[i][j].intervalue
                    self.grid[i][j].value = 1-self.grid[i][j].intervalue
                if(0<heuristic(self.grid[i][j],self.grid[d4i][d4j])<= drone.beam):
                    self.grid[i][j].intervalue += 1/heuristic(self.grid[i][j],self.grid[d4i][d4j])   
                    self.grid[i][j].signeed -= self.grid[i][j].intervalue
                    self.grid[i][j].value = 1-self.grid[i][j].intervalue
                if(self.grid[i][j].intervalue>0):
                    print(i,j,self.grid[i][j].intervalue)
    
    def GetSpotValue(self,i,j):
        return self.grid[i][j].value

    def SetPath(self,i,j):
        self.grid[i][j].path=True

    def GetPath(self,i,j):
        return self.grid[i][j].path

        #print(drone1.i,drone1.j)
        #print(drone2.i,drone2.j)

    def ResetView(self):
        # Putting the spot in the grid
        for i in range(self.size):
            for j in range(self.size):
                self.grid[i][j].path = False
        return self
  
    def ShowGrid(self):
        vis_grid = []

        self.grid[0][0].set = 80
        self.grid[self.size-1][self.size-1].set = 80

        for i in range(self.size):
            row = [0 for i in range(self.size)]
            vis_grid.append(row)

        for i in range(self.size):
            for j in range(self.size):
                if  self.grid[i][j].wall:
                    vis_grid[i][j] =  self.grid[i][j].set - 30
                elif self.grid[i][j].path:
                    vis_grid[i][j] =  80 
                elif self.grid[i][j].altitud:
                    vis_grid[i][j] =  self.grid[i][j].set - 10
                elif self.grid[i][j].crow:
                    vis_grid[i][j] = self.grid[i][j].set - 20
                elif self.grid[i][j].recharge:
                    vis_grid[i][j] = self.grid[i][j].set + 60
                elif self.grid[i][j].isdrone:
                    vis_grid[i][j] = self.grid[i][j].set + 30
                elif self.grid[i][j].interf:    
                    vis_grid[i][j] = self.grid[i][j].intervalue*30+20
                else:
                    vis_grid[i][j] = self.grid[i][j].set
        return vis_grid

    def action(self, choice):       #Gives us 8 total movement options. (0,1,2,3,4,5,6,7) 
        
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
             self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)
        elif choice == 4:
            self.move(x=1,y=0)
        elif choice == 5:
            self.move(x=0, y=1)
        elif choice == 6:
            self.move(x=-1, y=0)
        elif choice == 7:
            self.move(x=0, y=-1)
        
    def move(self, x=False, y=False):
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x
        if not y:
            self.y += np.random.randint(-1,2)
        else:
            self.y += y
            
        if self.x<0:
            self.x=0
        if self.x>=self.size:
            self.x = self.size-1
        if self.y<0:
            self.y=0
        if self.y>=self.size:
            self.y = self.size-1

def heuristic(a, b):                                   
    dist = math.sqrt((a.i - b.i)**2 + (a.j - b.j)**2)
    return dist

def ShowGraphic(weiba,weita,time,battery):

    weiba = weiba[::-1]
    weita = weita[::-1]
    battery = battery[::-1]
    time = time[::-1]

    print(weiba,battery,weita,time)
            
    plt.subplot(2,1,1)
    plt.title('Weight Tradeoff battery/time')
    plt.ylabel('Battery')
    plt.plot(weiba,battery)
            
    plt.subplot(2,1,2)
    plt.ylabel('Time')
    plt.xlabel('Weight')
    plt.plot(weita,time)