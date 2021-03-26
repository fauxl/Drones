import gym
import numpy as np
import random
from gym import spaces, logger
from gym.utils import seeding

from UtilityAgent import *
from spot import *

from Graphics import *

class DroneSignalEnv(gym.Env):
    """
    Description:
        A drone that has to reach a goal, in a grid environment with different spots,
        in order to save battery and don't take too much time at the same time.
    Source:
        Thesis project on drone path planning
    Observation:
        Type: Box(5)
        Num     Observation                 Min                      Max
        0       Drone i pos                izone                   size-1
        1       Drone j pos                jzone                   size-1
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
        Reward depends on the type of spot the drones moves on and on the action it has to perform, give signal or identify a fault
    Starting State:
        Drone i pos = izone
        Drone j pos = jzone
    Episode Termination:
        V Drone reached the goal
        X Time's up
        X Battery's dead
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,size):
        self.done = False
        self.starti = 0
        self.startj =0  
        self.state = None
        self.ok = 0
        self.size = size
        self.state_size = 3
        self.drone = Drone(1,1,1,self.starti,self.startj,0)

        self.grid = GraphEnv(self.size,[])
        self.grid.PutDrones()

        low = np.array([0,0])
        high = np.array([19,19])

        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.action_space = spaces.Discrete(8)
       # self.observation_state = spaces.Box(low, high, dtype=np.float32)

    def outbounds(self,i,j):
        if(i>self.size-1 or j>self.size-1 or i<0 or j<0):
            return True
        else:
            return False

    def checkwall(self,i,j):
        if self.grid.grid[i][j].wall==True:
            return True
    
    def checkrecharge(self,i,j):
        if self.grid.grid[i][j].recharge==True:
            return True

    def checkSignal(self,i,j):
        if 0.1<self.grid.grid[i][j].intervalue<=0.31:
            return True
    
    def checkSignal2(self,i,j):
        if self.grid.grid[i][j].intervalue>=0.33:
            return True

    def checkSignal3(self,i,j):
        if 0.31<self.grid.grid[i][j].intervalue<=0.329:
            return True

    def evalue(self,i,j):

        reward = -1

        return reward

    def try_step(self, action):

        pos = self.drone.getPos()
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

        
        return newi, newj

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
        end = True

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

        
        if not self.outbounds(newi,newj):

            self.drone.move(newi,newj,self.drone.battery,self.drone.time,self.drone.distance)
            self.state = (newi,newj)
            #print("hello") 
            
            if self.grid.GetPath(newi,newj) and self.checkSignal3(newi,newj):
                reward=0.5


            elif self.checkwall(newi,newj):
                self.drone.move(pos[0],pos[1],self.drone.battery,self.drone.time,self.drone.distance)
                self.state = (pos[0],pos[1])

                reward = -2
                
            elif self.checkSignal(newi,newj):
                self.grid.SetPath(newi,newj)
                reward = 0

            elif self.checkSignal3(newi,newj):
                self.grid.SetPath(newi,newj)
                self.ok =1
                reward = 2

            elif self.checkSignal2(newi,newj):
                self.grid.SetPath(newi,newj)
                reward = -1

            else: 
                #self.grid.SetPath(newi,newj)
                self.grid.SetPath(newi,newj)
                reward = -2
                if self.ok == 1:
                    reward=-2
                    self.done=True


            observation = np.array(self.state)


            """
            elif self.checkSignal2(newi,newj):
                self.grid.SetPath(newi,newj)
                self.drone.move(newi,newj,self.drone.battery,self.drone.time,self.drone.distance)
                self.state = (newi,newj)
                observation = np.array(self.state)
                reward = 0   
            """

        else:  

            self.drone.move(pos[0],pos[1],self.drone.battery,self.drone.time,self.drone.distance)
            self.state = (newi,newj)
            observation = np.array(self.state)
            reward = -10
            print(newi,newj)
            self.done = True

        


        for i in range(self.size):
            for j  in range(self.size):
                if self.checkSignal3(i,j):
                    if not self.grid.GetPath(i,j):
                        #print(i,j)
                        end = False
                      
        if (end):
            self.done = True
            info  = 1
            print ('win')

        """
        if (newi==10 and newj==18):
            info =2
            self.ok =1
            reward = 10
            self.done=True
            #print ('wino')
        """
        return observation, reward, self.done, info
        
    def reset(self):      
        self.done = False
        self.drone = Drone(1,1,1,self.starti,self.startj,0)
        low = np.array([0,0])
        high = np.array([19,19])
        self.ok = 0

        self.observation_state = spaces.Box(np.float32(low), np.float32(high), dtype=np.float32)
        self.state = (self.starti,self.startj)
        observation = np.array(self.state)

        self.grid = GraphEnv.ResetView(self.grid)

        return np.array(self.state)

    def render(self, mode='human'):

        filename = 'img/DroneInterference.png'

        fig = plt.figure(figsize=(11, 7))
        timer = fig.canvas.new_timer(interval = 300) #creating a timer 
        timer.add_callback(close_event)
        plt.title("Griglia")
        #print("we")
        plt.imshow(self.grid.ShowGrid())
        plt.axis("off")
        plt.savefig(filename)
        plt.show()

        #timer.start()

    def get_action(self, state):
        action = self.step(random.choice(range(self.action_space.n)))

    def _take_action(self, action):
        azione = self.step(action)      
        print (azione)
    
def close_event():
    plt.close() #timer calls this function after 3 seconds and closes the window 