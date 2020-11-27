from lxml import etree
import matplotlib.pyplot as plt
import numpy as np
import random
import math

#import Route

"""
Spots type on the map	(DECIDE)       Normalized Utility Scores Base         Probability of presence on the map

- Recharge Station  				    0.9                                 0.1                   
- Walkable spots    				    1    			        0.4
- Minimum altitude specification	   	   0.7                                  0.15
- Interferences zones       			   0.3                                  0.05
- Crowded zones 				   0.5                                  0.1
- Obstacles/No fly zones                            0                           0.2     Utility is equal to 0 can be ignored


Drone Battery       		       Normalized Utility Scores          

From 100% to 5%  			        0 to 1		                Each block percentage down by one                     
if less then 5 then failed		      			
 		   	                         

Time Elapsed           		       Normalized Utility Scores            

From 0 to 85 min     			        0 to 1			         Each block time up 1 min
if more then 85 then failed         		   	                        


Distance from target (A*)

From 0 to max length                            0 to 1                            Estimated with an euristich function  A*

"""

"utility obtained by "

# Utility Function 
class UtilityAgent:
        ws = 0.9
        rs = 1
        mas = 0.7
        cwz = 0.5
        inz = 0.3 

        pws = 0.4
        prs = 0.1
        pmas = 0.15
        pcwz = 0.1
        pinz = 0.05 

        eus = (ws*pws) + (rs*prs) + (mas*pmas) + (cwz*pcwz) + (inz*pinz)

        #print(eus)

      
class Drone:
        def __init__(self,battery,time,distance):
                self.i = 0
                self.j = 0
                self.battery = battery
                self.time = time
                self.distance = distance #per funzione di ottimizazzione parte da 1 e arriva a 0
        
        def move(self,i,j,battery, time, distance):
                self.i = i
                self.j = j
                self.battery = battery
                self.time = time
                self.distance = distance
                #print(self.i,self.j,self.battery,self.time,self.distance)

def Weight(drone):
        weiba = (2-drone.battery)*0.35
        weita = (2-drone.time)*0.35

        weiba = round(((0.7*weiba)/(weiba+weita)),2)
        #print(weiba)
        #weiba = weiba + round(0.01*((2-drone.battery)),2)
        weita = round(0.7-weiba,2)
        
        #print(weiba,weita)
        return weiba, weita

def UtilityFunc(kind,battery,time,distance,weight):
        #print(weight[0],weight[1],time,battery,kind)
        uf = (weight[0]*battery + weight[1]*time+ 0.30*distance + kind)
        return uf


"""
1. Imposta le variabili di prova per lo stato corrente.
2. Per ogni possibile valore del nodo di decisione:
a. assegna tale valore al nodo di decisione;
b. calcola le probabilità a posteriori dei nodi genitori del nodo di
utilità con un algoritmo standard di inferenza probabilistica;
c. calcola l’utilità risultante dell’azione.
3. Restituisci l’azione con utilità più alta.


int node = #a seconda del valore asssegnato si indica una delle tipologie di celle
 probgen
utility
start = True
       while (start) {
               if (utility<newutility)
                        utility = newutility 
       } 


"""""