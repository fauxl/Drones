import gym
import numpy as np
import random
from gym import spaces, logger
from gym.utils import seeding

from UtilityAgent import *
from spot import *

from Graphics import *

class DroneEnv(gym.Env):
    """
    Description:
        A drone that has to reach a goal, in a grid environment with different spots,
        in order to save battery and don't take too much time at the same time.
    Source:
        Thesis project on drone path planning
    Observation:
        Type: Box(5)
        Num     Observation               Min                    Max
        0       Battery                     0                     1
        1       Drone i pos                 0                    size-1
        2       Drone j pos                 0                    size-1
        3       Time                        0                       1
        4       Distance                    0                    func calc
    Actions:
        Type: Discrete(8)
        Num   Action
        0     Drone goes right
        1     Drone goes left
        2     Drone goes up
        3     Drone goes down
        4     Drone goes up-right
        5     Drone goes up-left
        6     Drone goes down-right
        7     Drone goes down-left
    Reward:
        Reward depends on the type of spot the drones moves on and the change in his battery and in time, distance
    Starting State:
        Time=1
        Battery=1
        Distance = MAX
        Drone i pos = 0
        Drone j pos = 0
    Episode Termination:
        V Drone reached the goal
        X Time's up
        X Battery's dead
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,size):
        self.done = False
        self.state = None
        self.drone = Drone(1,1,1,0,0,0)
        self.size = size
        self.state_size = 5
        
        self.grid = GraphEnv(self.size,[])
        #self.grid.PutDrones()
        #GraphEnv.SetDroneSignal(self.grid)

        self.startdis = self.grid.heuristic(0,0,self.size-1,self.size-1)

        low = np.array([0,0,0,0,0])
        high = np.array([1,self.size-1,self.size-1,1,1])

        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.action_space = spaces.Discrete(8)
       # self.observation_state = spaces.Box(low, high, dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def outbounds(self,i,j):
        if(i>self.size-1 or j>self.size-1):
            return True
        elif(i<0 or j<0):
            return True
        else:
            return False

    def checkover(self):
        if(self.drone.battery<0.03 or self.drone.time <0.03):
            return True
        else:
            return False

    def checkSignal(self,i,j):
        if self.grid.grid[i][j].intervalue>0.3:
            return True 

    def checkwall(self,i,j):
        if self.grid.grid[i][j].wall==True:
            return True
    
    def checkrecharge(self,i,j):
        if self.grid.grid[i][j].recharge==True:
            return True

    def evalue(self,i,j):
    
        pos = self.drone.getPos()
            
        battery =  round(self.drone.battery- 0.01,2)
        time =  round(self.drone.time- 0.004,3)

        dis = self.grid.heuristic(i,j,self.size-1,self.size-1)
        dis= round(dis/self.startdis,2)
        distance = 2-dis

        #print(distance)
        weight= Weight(self.drone.battery,self.drone.time)
        #print(distance)
        #reward = self.grid.GetSpotValue(i,j)*distance
    
        reward =   UtilityFunc(self.grid.grid[i][j].value,battery,time,distance,weight)

        return battery,time, distance, reward


    def step(self, action):

        """
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning. 
        """
        pos = self.drone.getPos()
        val = ""   
        info = 0
        newi = pos[0]
        newj = pos[1]


        if (action==0):
                    
            newi = pos[0]
            newj = pos[1]+1

        elif (action == 1):

            newi = pos[0]
            newj = pos[1]-1

        elif (action == 2):

            newi = pos[0]-1
            newj = pos[1]

        elif (action == 3):

            newi = pos[0]+1
            newj = pos[1]

        elif (action == 4):

            newi = pos[0]-1
            newj = pos[1]+1

        elif (action == 5):

            newi = pos[0]-1
            newj = pos[1]-1

        elif (action == 6):   
                
            newi = pos[0]+1
            newj = pos[1]+1

        elif (action == 7):      
                
            newi = pos[0]+1
            newj = pos[1]-1

        self.state = (self.drone.battery,pos[0],pos[1],self.drone.time,self.drone.distance)

        #check if  next spot is in the grid 
        if not self.outbounds(newi,newj):  

            #compute the new values of battery, time, distance and reward
            val = self.evalue(newi,newj)

            #check the kind of spot and change reward and state accordingly
            if self.checkrecharge(newi,newj):
                reward = 0
                weight= Weight(self.drone.battery,self.drone.time)
                if(weight[0]>weight[1] and weight[0]>=0.332):
                    battery =  1
                    time = round(self.drone.time - 0.05,2)
                    self.drone.move(newi,newj,battery,time,val[2])
                    self.state = (battery,newi,newj,time,val[2])
                    reward = 80*self.grid.grid[newi][newj].important
                #print(reward,weight[0])

            elif self.checkwall(newi,newj):
                reward = 0
                self.done = False

            else:
                #print("hegh") 
                self.drone.move(newi,newj,val[0],val[1],val[2])
                self.state = (val[0],newi,newj,val[1],val[2])
                self.grid.SetPath(newi,newj)

                reward = val[3]*self.grid.grid[newi][newj].important
                #print(newi,newj)

            #check if the drone is arrived at the goal or the battery/time is elapsed
            if  self.checkover():
                self.done = True
                info = -1
                reward = -5*self.grid.grid[newi][newj].important
                #print(" ",self.drone.battery,self.drone.time)

            elif (newi == self.size-1 == newj):
                self.done = True
                reward = (100+10*self.drone.battery)*self.grid.grid[newi][newj].important
                info = 1
                print("\nDone",self.drone.battery,self.drone.time,self.drone.i,self.drone.j)
               
               
                for i in range (self.size):
                    for j in range (self.size):
                        if self.grid.GetPath(i,j):
                            self.grid.grid[i][j].important += 1

        else:
            #print("hello") 
            #print(newi,newj)
            reward = 0
            self.done = False

        #check observation with the new state
        observation = np.array(self.state)
        
        return observation, reward, self.done, info

    #reset the grid for the next execution
    def reset(self):      
        self.done = False
        self.drone = Drone(1,1,1,0,0,0)
        low = np.array([0,0,0,0,0])
        high = np.array([1,self.size-1,self.size-1,1,1])

        self.observation_state = spaces.Box(np.float32(low), np.float32(high), dtype=np.float32)
        self.state = (1,0,0,1,1)
        observation = np.array(self.state)

        self.grid = GraphEnv.ResetView(self.grid)

        return np.array(self.state)

    #render the grid
    def render(self, mode='human'):

        fig = plt.figure(figsize=(11, 7))
        timer = fig.canvas.new_timer(interval = 300) #creating a timer 
        timer.add_callback(close_event)
        plt.title("Griglia")
        #print("we")
        plt.imshow(self.grid.ShowGrid())
        plt.axis("off")

        #timer.start()
        plt.show()
        
  
    def get_action(self, state):
        action = random.choice(range(self.action_space.n))
        print (action)

    def _take_action(self, action):
        azione = self.step(action)      
        print (azione)
    
def close_event():
    plt.close() #timer calls this function after 3 seconds and closes the window 