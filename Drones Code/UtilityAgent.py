from lxml import etree
import matplotlib.pyplot as plt
import numpy as np
import random
import math

"""
Spots type on the map		       Normalized Utility Scores            Probability

- Recharge Station  				            1                           0.1                   
- Walkable spots    				           0.9			                0.4
- Minimum altitude specification	   	       0.7                          0.15
- Interferences zones       			       0.3                          0.05
- Crowded zones 				               0.5                          0.1
- Obstacles/No fly zones                        0                           0.2     Utility is equal to 0 can be ignored

Drone Battery       		       Normalized Utility Scores            Probability

- Battery above or equal 70%  				        1                       0.33                  
- Battery under or equal 30%    				   0.3			            0.33
- Battery between 70% -  30%    		   	       0.6                      0.34

Time Elapsed           		       Normalized Utility Scores            Probability

- Started from 15 min        				        1                       0.33                  
- Started from 30 min              				   0.6			            0.33
- Started from 45 min           		   	       0.3                      0.34


"""


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

        print(eus)

        b70 = 1
        b30 = 0.3
        b37 = 0.6

        pb70 = 0.33
        pb30 = 0.33
        pb37 = 0.34

        eub = (b70*pb70) + (b30*pb30) + (b37*pb37) 

        print(eub)

        t15 = 1
        t30 = 0.6
        t45 = 0.3

        pt15 = 0.33
        pt30 = 0.33
        pt45 = 0.34

        eut = (t15*pt15) + (t30*pt30) + (t45*pt45) 

        print(eut)



        


