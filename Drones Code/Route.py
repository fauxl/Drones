from lxml import etree
import matplotlib.pyplot as plt
import numpy as np
import random
import math

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
        self.set = 0        # For coloring purpose on the graph
        self.i = i          
        self.j = j
        self.f = 0      # f is the cost function f = g+h
        self.g = 0      # g is like the actual distance from the starting point to the node we are
        self.h = 0      # h is the heuristic - estimates the distance from the node we are to the end for each possibile node - here it has been used euclidean distance

        self.parent = None          # To trace back the path, we need to record parents
        self.wall = False           # spot obstacle set false
        self.neighbors = []         # list of neighbors of a spot              
       
        if random.random() < 0.4:   # Percentage of obstacles or not linked spots generated randomly
            self.wall = True

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

size = 50
grid = []
openSet = []
closedSet = []

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

start = grid[0][0]                  # Start can be randomized = grid[int(random.random()*40)][int(random.random()*40)]
end = grid[size - 1][size - 1]      # Destination can be randomized = grid[int(random.random()*40)][int(random.random()*40)]
grid[0][0].wall = False
grid[size - 1][size - 1].wall = False       

# Adding the start point to the open_set
openSet.append(start)

loop = True
while loop:
    if len(openSet) > 0:
        winner = 0
        for i in range(len(openSet)):
            if openSet[i].f < openSet[winner].f:
                winner = i

        current = openSet[winner]

        # Path found
        if current == end:
            current.set = 8
            temp = current.f
            while current.parent:
                current.parent.set = 16
                current = current.parent
            print('The program finished, the shortest distance \n to the path is ' + str(temp) + ' blocks away \n')
            loop = False

        # Remove the evaluated point from open_set and add to closed set
        openSet.pop(winner)
        closedSet.append(current)

        # Adding a new point to evaluate in open_set from neighbors
        neighbors = current.neighbors
        for neighbor in neighbors:
            if neighbor not in closedSet:
                if not neighbor.wall:
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
                        neighbor.parent = current

    # Path not Found
    else:
        current.set = 12
        while current.parent:
            current.parent.set = 8
            current = current.parent
        print('No path found!')
        loop = False

# Visualization
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
plt.imshow(vis_grid)
plt.show()

