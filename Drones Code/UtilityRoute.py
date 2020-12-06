import matplotlib.pyplot as plt
import numpy as np
import random
import math
import matplotlib as mpl

from UtilityAgent import *

from Graphics import *

# Usually data model for maps is G(N,E) 
# N are the nodes, in this case ID
# E are the edges that link the N
# We have to find a representation suitable
# Most common one is using a grid with obstacles

# Working on map data import

"""PostX = []                                              
PostY= []
PostName = []
RoadId = []
tree = etree.parse("XMLFILE.xml")

for node in tree.xpath("/osm/node"):
    PostX.append(node.attrib['lat'])  
    PostY.append(node.attrib['lon'])  
    PostName.append(node.attrib['id'])  

for node in tree.xpath("/osm/way"):
    RoadId.append(int(node.attrib['id']))  
    for elements in tree.xpath("/osm/way"):
        if node.attrib['id']==elements.attrib['id']
            RoadPart

print (RoadId)
plt.show()
"""


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

        self.parent = None          # To trace back the path, we need to record parents
        self.wall = False           # spot obstacle set false
        self.crow = False
        self.recharge = False
        self.interf = False
        self.altitud = False

        self.neighbors = []         # list of neighbors of a spot              
       
        if random.random() < 0.001:   # Percentage of obstacles or not linked spots generated randomly
            if self.value==1:
                self.recharge = True
                self.value=0.9

        if random.random() < 0.1:   # Percentage of obstacles or not linked spots generated randomly
            if self.value==1:
                self.altitud= True
                self.value=0.7

        if random.random() < 0.099:   # Percentage of obstacles or not linked spots generated randomly
            if self.value==1:
                self.interf= True
                self.value=0.3

        if random.random() < 0.2:   # Percentage of obstacles or not linked spots generated randomly
            if self.value==1:
                self.wall = True
                self.value=0

        if random.random() < 0.1:   # Percentage of obstacles or not linked spots generated randomly
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

size = 100
grid = []
openSet = []
closedSet = []
BatteryValues = []
TimeValues = []
WeibaValues = []
WeitaValues = []


# Heuristic Euclidean Distance
def heuristic(a, b):                                   
    dist = math.sqrt((a.i - b.i)**2 + (a.j - b.j)**2)
    return dist

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

        
start = grid[0][0]  # Start can be randomized = grid[int(random.random()*40)][int(random.random()*40)]
end = grid[size - 1][size - 1]      # Destination can be randomized = grid[int(random.random()*40)][int(random.random()*40)]
grid[0][0].wall = False
grid[0][0].altitud = False
grid[0][0].crow= False
grid[0][0].interf = False
grid[0][0].recharge = False

grid[size - 1][size - 1].wall = False       
grid[size - 1][size - 1].altitud = False
grid[size - 1][size - 1].crow= False
grid[size - 1][size - 1].interf = False
grid[size - 1][size - 1].recharge = False

# Adding the start point to the open_set
openSet.append(start)
start.f = heuristic(start, end)
drone = Drone(1,1,1)

"""
# Visualization debugging
vis_grid = []
for i in range(size):
    row = [0 for i in range(size)]
    vis_grid.append(row)

start.set = 20
end.set = 25
for i in range(size):
    for j in range(size):
        if grid[i][j].wall:
            vis_grid[i][j] = grid[i][j].set - 10
        else:
            vis_grid[i][j] = grid[i][j].set

plt.figure(figsize =(8, 7))
plt.title('A* Algorithm - Shortest Path Finder\n')
plt.ion()
im = plt.imshow(vis_grid)
"""

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
                #print(current.tim)
                drone.move(current.i,current.j,current.bat,current.tim,current.f)
                #print(current.utility)
               
                # Path found
                if current == end:
                    current.set = 8
                    temp = current.f
                    while current.parent:
                        current.parent.set = 16
                        WeibaValues.append(current.weiba)
                        WeitaValues.append(current.weita)
                        TimeValues.append(current.tim)
                        print(current.value)
                        BatteryValues.append(current.bat)
                        current = current.parent
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
                        if not neighbor.wall :

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
        if grid[i][j].wall:
            vis_grid[i][j] =  grid[i][j].set - 30
        elif grid[i][j].altitud:
            vis_grid[i][j] =  grid[i][j].set - 10
        elif grid[i][j].crow:
            vis_grid[i][j] = grid[i][j].set - 20
        elif grid[i][j].interf:
            vis_grid[i][j] = grid[i][j].set - 40
        elif grid[i][j].recharge:
            vis_grid[i][j] = grid[i][j].set + 60
        else:
            vis_grid[i][j] = grid[i][j].set

plt.figure(figsize =(8, 7))
plt.title('A* Algorithm with Utility - Shortest Path Finder\n')
plt.imshow(vis_grid)
plt.show()
