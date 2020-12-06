import matplotlib.pyplot as plt
import numpy as np

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
    