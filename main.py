
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
#----------------------   Defining Global variables od vision system -------

#print(myRobot.PoseAr)
logging.basicConfig(level=logging.INFO)
shapeMask =0
last = time.time()
smoothing = 0.9
DepthScale = 125
fps_smooth = 60
ObjectsGridNumber = 20
ObjectsSamples  = 7
ObjectsOrientationInGrid = np.zeros((ObjectsSamples,ObjectsGridNumber,ObjectsGridNumber),dtype='int16')
ObjectsLabelInGrid       = np.zeros((ObjectsSamples,ObjectsGridNumber,ObjectsGridNumber),dtype='int16')
kernel2=np.array([[1,  1,  1, 1, 1],
                  [1,  1, -2, 1, 1],
                  [1, -2,-12,-2, 1],
                  [1,  1, -2, 1, 1],
                  [1,  1,  1, 1, 1]
                 ])
kernel1=np.array([[1,  1, 1],
                 [1,  -5, 1],
                 [1,  1, 1]
                 ])

ObjectsLabel= { 0000: ' -------   ',
                1020: 'Bk SmProfl ',
                1021: 'Wh SmProfl ',
                1040: 'Bk BgProfl ',
                1041: 'Wh BgProfl ',
                1120: 'Screw      ',
                1220: 'Sm Nut     ',
                1230: 'Bg Nut     ',
                1320: 'PlasticTube',
                1410: 'Bearing Box',
                1510: 'Bearing    ',
                1610: 'Axis       ',
                1710: 'Dist  Tube ',
                1810: 'Motor      ',
                9999: 'Nothings   ' }

ImageSize             = (640  , 480)
ImageCenter           = (ImageSize[0]/2, ImageSize[1]/2)
Dthreshold            = 120
MinArea               = 100
MaxArea               = 50000
Wcrop                 = 15
kernel                = np.ones((5, 5), np.float32) / 15
surf                  = cv2.SURF(85)
surf.hessianThreshold =  1150
surf.extended         = 0     # 0 for 64 Bins , 1 for 128 Bins
BoWSize               = 100
svm_params            = dict(kernel_type = cv2.SVM_RBF, svm_type = cv2.SVM_C_SVC, C=4.67, gamma=1.383)
DepthSamples          = 4
DDepths = np.zeros(shape=(ImageSize[1],ImageSize[0]) + (DepthSamples+1,))
ImgCADs = np.zeros(shape=(ImageSize[1],ImageSize[0]) + (DepthSamples+1,))
DepthM  =0
#===========================================================================

#===========================================================================
#------------------- Defining basic functins -------------------------------
#        These functions are calls in intialization and in the main loop
#---------------------------------------------------------------------------
