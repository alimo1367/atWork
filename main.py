
#from geometry_msgs.msg import Twist
import numpy as np
import glob
import pickle

import logging
import time
import cv2
import pyrealsense as pyrs
import msgpackrpc
#from pyrealsense import rsutilwrapper
#from ipdb3 import set_trace



#print(cv2.__version__)


from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
#from matplotlib import pyplot as plt
from constants import rs_option
#from blendfunc import blend

#------------------------------------------------------
#----------        Threading for send data ------------
import Pyro4
import thread


@Pyro4.expose

class GreetingMaker2(object):
    def get_fortune(self, name):
        global Idx, Dx,Dy,Dp,Orint,Width,Hight
        if(name==0):
          Dx, Dy, Dp, Orint ,Width,Hight= 0,0,0,0,0,0
        return Idx,Dx,Dy,Dp,Orint,str(Width),str(Hight)

class SumServer(object):
    def get_fortune(self, name):
        global Idx, Dx,Dy,Dp,Orint,Width,Hight
        if(name==0):
          Dx, Dy, Dp, Orint ,Width,Hight= 0,0,0,0,0,0
        return Idx,Dx,Dy,Dp,Orint,str(Width),str(Hight)



def run1(SS):
    global daemon
    # while True:
    print("Ready.")
    time.sleep(SS)
    daemon.requestLoop()  # start the event loop of the server to wait for calls









#-----------------------------
SS=1
thread.start_new_thread(run1, (SS,))

daemon = Pyro4.Daemon()  # make a Pyro daemon
ns = Pyro4.locateNS()  # find the name server
uri = daemon.register(GreetingMaker2)  # register the greeting maker as a Pyro object
ns.register("ali1", uri)  # register the object with a name in the name server
#------------------------------------------------------

#===========================================================================
