import matplotlib.pyplot as plt
import tensorflow as tf

import random
import math


class Drone:
        def __init__(self,battery,distance):
                self.i = 0
                self.j = 0
                self.battery = battery
                #self.time = time
                self.distance = distance
                self.signalp = 0
        
        def move(self,i,j,battery,distance):
                self.i = i
                self.j = j
                self.battery = battery
                #self.time = time
                self.distance = distance
                self.signalp = 0
      
               
                #print(self.i,self.j,self.battery,self.time,self.distance)
