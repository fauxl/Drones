
import matplotlib.pyplot as plt
import numpy as np
import random
import math

from UtilityAgent import *
from Graphics import *

# Usually data model for maps is G(N,E) 
# N are the nodes, in this case ID
# E are the edges that link the N
# We have to find a representation suitable
# Most common one is using a grid with obstacles

# Working on map data import

# A* alghoritm first version, to be optimized and revised

# Creating a class 'spot' which would be like a node on the screen
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
       
        if random.random() < 0.002:   # Percentage of recharge zone
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
    def add_neighbors(self, grid_passed):
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

def SetSignal(grid,d1,beam):
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

size = 100
grid = []
openSet = []
distance = 8
closedSet = []
BatteryValues = []
TimeValues = []
WeibaValues = []
WeitaValues = []

d1i = random.randint(20, 70)
d1j = random.randint(20, 80)

d2i = d1i + distance
d2j = d1j + distance

d3i = d1i + distance
d3j = d1j - distance

d4i =  d1i + distance*2
d4j = d1j 

drone = Drone(1,1,1,0,0,7)
drone1 = Drone(1,1,1,d1i,d1j,7)
drone2 = Drone(1,1,1,d2i,d2j,7)
drone3 = Drone(1,1,1,d3i,d3j,7)

# Heuristic Euclidean Distance
def heuristic(a, b):                                   
    dist = math.sqrt((a.i - b.i)**2 + (a.j - b.j)**2)
    return dist

def checkInter(grid):
    if(grid[i][j].interavalue>0):
        return True


# Setting drone signal based on distance and setting value and signal need of the spots covered by it
def Distance(grid,i,j):
    if(0<heuristic(grid[i][j],grid[d1i][d1j])<= drone1.beam):
        grid[i][j].intervalue = grid[i][j].intervalue + 1/heuristic(grid[i][j],grid[d1i][d1j])
        grid[i][j].signeed -= grid[i][j].intervalue
        grid[i][j].value = 1-grid[i][j].intervalue
    if(0<heuristic(grid[i][j],grid[d2i][d2j])<= drone2.beam):
        grid[i][j].intervalue = grid[i][j].intervalue +1/heuristic(grid[i][j],grid[d2i][d2j])
        grid[i][j].signeed -= grid[i][j].intervalue
        grid[i][j].value = 1-grid[i][j].intervalue
    if(0<heuristic(grid[i][j],grid[d3i][d3j])<= drone3.beam):
        grid[i][j].intervalue= grid[i][j].intervalue +1/heuristic(grid[i][j],grid[d3i][d3j])   
        grid[i][j].signeed -= grid[i][j].intervalue
        grid[i][j].value = 1-grid[i][j].intervalue
    if(0<heuristic(grid[i][j],grid[d4i][d4j])<= drone.beam):
        grid[i][j].intervalue = grid[i][j].intervalue +1/heuristic(grid[i][j],grid[d4i][d4j])   
        grid[i][j].signeed -= grid[i][j].intervalue
        grid[i][j].value = 1-grid[i][j].intervalue

# Making a grid
for i in range(size):
    row = [spot(0, 0) for i in range(size)]
    grid.append(row)

# Putting the spot in the grid
for i in range(size):
    for j in range(size):
        grid[i][j] = spot(i, j)

# Filling neighbours
for i in range(size):
    for j in range(size):
        grid[i][j].add_neighbors(grid)

SetSignal(grid,drone1,drone1.beam)
SetSignal(grid,drone2,drone2.beam)
SetSignal(grid,drone3,drone3.beam)
grid[drone1.i][drone1.j].isdrone=True
grid[drone2.i][drone2.j].isdrone=True
grid[drone3.i][drone3.j].isdrone=True

# Filling Network
for i in range(size):
    for j in range(size):
        Distance(grid,i,j)
        #print(grid[i][j].signeed)


start = grid[0][0]                  # Start can be randomized = grid[int(random.random()*40)][int(random.random()*40)]
end = grid[d4i][d4j]      # Destination can be randomized = grid[int(random.random()*40)][int(random.random()*40)]
grid[0][0].wall = False
grid[size - 1][size - 1].wall = False       

# Adding the start point to the open_set
openSet.append(start)
start.f = heuristic(start, end)

loop = True
while loop:
    if (drone.battery>0 and drone.time>0):
            if len(openSet) > 0:
                winner = 0    
                for p in range(len(openSet)):
                    if openSet[p].utility > openSet[winner].utility:
                        winner = p
                        
                        #print(openSet[winner].recharge,openSet[winner].altitud,openSet[winner].wall,openSet[winner].crow,openSet[winner].interf) 
                        #vis_grid[drone.i][drone.j]=4.6
                        #im.set_data(vis_grid)
                        #plt.pause(0.000001)
                        #plt.draw()       

                current = openSet[winner]
                drone.move(current.i,current.j,current.bat,current.tim,current.f)
               
                # Path found
                if current == end:
                    grid[i][j].isdrone == True
                    current.set = 8
                    temp = current.f
                    while current.parent:
                        current.parent.set = 60
                        WeibaValues.append(current.weiba)
                        WeitaValues.append(current.weita)
                        TimeValues.append(current.tim)
                        BatteryValues.append(current.bat)
                        current = current.parent
                    SetSignal(grid,drone,drone.beam)
                    print('The program finished, the length of the path is ' + str(round(temp*start.f,0)) + ' blocks away \n')
                    print('The program finished, the Drone battery is '+ str(drone.battery)  + '\n the time remaining is ' + str(drone.time))

                    loop = False
                    ShowGraphic(WeibaValues,WeitaValues,TimeValues,BatteryValues)

                # Remove the evaluated point from open_set and add to closed set
                openSet.pop(winner)
                closedSet.append(current)

                # Adding a new point to evaluate in open_set from neighbors
                neighbors = current.neighbors
                for neighbor in neighbors:
                    if neighbor not in closedSet:
                        if not neighbor.wall and not neighbor.interf :

                            temp_g = current.g + 1              
                            new_path = False

                            if neighbor in openSet:
                                if temp_g < neighbor.g:
                                    neighbor.g = temp_g    
                                    new_path = True
                            
                            else:
                                neighbor.g = temp_g
                                new_path = True
                                openSet.append(neighbor)
                             
                            if new_path:
                                neighbor.h = heuristic(neighbor, end)
                                neighbor.f = neighbor.g + neighbor.h
                                NormDis = round(neighbor.f/start.f,2)
                                neighbor.f = 2-NormDis
                                
                                neighbor.parent = current    
                                parne = neighbor.parent
                                curbat = parne.bat
                                curtime = parne.tim 

                                neighbor.bat = round(curbat - 0.01,2)
                                neighbor.tim = round((curtime - 0.004),3)
                            
                                if neighbor.recharge:
                                
                                    neighbor.bat = 1
                                    neighbor.tim = round(curtime - 0.05,2)

                                weight= Weight(neighbor.bat,neighbor.tim)
                                neighbor.weiba = round(weight[0],2)
                                neighbor.weita = round(weight[1],2)

                                neighbor.utility = UtilityFunc(neighbor.value,neighbor.bat,neighbor.tim,neighbor.f,weight)

            # Path not Found
            else:
                current.set = 12
                while current.parent:
                    current.parent.set = 8
                    current = current.parent
                print('No path found!')    
                loop = False
    
    else:
        loop = False
        print('No path found!, the Drone battery is '+ str(drone.battery)  + '\n the time elapsed is ' + str(1-drone.time)) 
    

# Visualization
vis_grid = []
for i in range(size):
    row = [0 for i in range(size)]
    vis_grid.append(row)

start.set = 80
end.set = 80
for i in range(size):
    for j in range(size):
        vis_grid[i][j] = grid[i][j].set
        if grid[i][j].recharge:
                vis_grid[i][j] = grid[i][j].set + 60
        if grid[i][j].set != 60:
            if grid[i][j].wall:
                vis_grid[i][j] =  grid[i][j].set - 30
            elif grid[i][j].altitud:
                vis_grid[i][j] =  grid[i][j].set - 10
            elif grid[i][j].crow:
                vis_grid[i][j] = grid[i][j].set - 20
            elif grid[i][j].isdrone:
                vis_grid[i][j] = grid[i][j].set + 30
            elif grid[i][j].interf:    
                vis_grid[i][j] = grid[i][j].intervalue*30+20

#Save  
filename = 'img/UtilityAlgorithmwithAStar.png'
plt.figure(figsize =(8, 7))
plt.title('Utility Algorithm with A*\n')
plt.axis("off")        

plt.imshow(vis_grid)
plt.savefig(filename)
plt.show()

