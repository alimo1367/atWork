
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
def CamInit():
    global cam,c,d,Rows,Cols,N,Xc,Yc
    global P11,P12,P13,P21,P22,P23,P31,P32,P33
    global extrinsics
    global depth100 , cad100 , dac100 , depthscale100
    #pyrs.start()
    serv = pyrs.Service()

    cam = serv.Device(device_id=0, streams=[pyrs.stream.CADStream(fps=60),
                                            pyrs.stream.DACStream(fps=60),
                                            pyrs.stream.ColorStream(fps=60),
                                            pyrs.stream.DepthStream(fps=60)])
    #extrinsics = cam.get_device_extrinsics(cam.streams[1].stream, cam.streams[0].stream)
    cam.wait_for_frames()
    c = cam.color
    depth100 = cam.depth
    cad100 = cam.cad
    dac100 = cam.dac
    depthscale100 = cam.depth_scale

    #DepthM = cam.depth * cam.depth_scale * 1000
    DepthM = np.uint8(cam.dac * cam.depth_scale * 125)
    Rows, Cols = DepthM.shape[:2]
    N = 100
    Xc = Rows / 2
    Yc = Cols / 2

    P11 = [Xc - N, Yc - N]
    P12 = [Xc - 0, Yc - N]
    P13 = [Xc + N, Yc - N]
    P21 = [Xc - N, Yc + 0]
    P22 = [Xc - 0, Yc + 0]
    P23 = [Xc + N, Yc + 0]
    P31 = [Xc - N, Yc + N]
    P32 = [Xc - 0, Yc + N]
    P33 = [Xc + N, Yc + N]

#---------------------------------------------------------------------------
def Cam_Config():
    global cam
    pyrs.core.DeviceBase.set_device_option(cam,option=rs_option.RS_OPTION_COLOR_ENABLE_AUTO_WHITE_BALANCE,value=-1)
    pyrs.core.DeviceBase.set_device_option(cam,option=rs_option.RS_OPTION_COLOR_BACKLIGHT_COMPENSATION,value=0)
    pyrs.core.DeviceBase.set_device_option(cam,option=rs_option.RS_OPTION_COLOR_BRIGHTNESS,value=10)
    pyrs.core.DeviceBase.set_device_option(cam,option=rs_option.RS_OPTION_COLOR_CONTRAST,value=50)
    pyrs.core.DeviceBase.set_device_option(cam,option=rs_option.RS_OPTION_COLOR_EXPOSURE,value=500)
    pyrs.core.DeviceBase.set_device_option(cam,option=rs_option.RS_OPTION_COLOR_GAIN,value=65)
    pyrs.core.DeviceBase.set_device_option(cam,option=rs_option.RS_OPTION_COLOR_GAMMA,value=100)
    pyrs.core.DeviceBase.set_device_option(cam,option=rs_option.RS_OPTION_COLOR_HUE,value=0)
    pyrs.core.DeviceBase.set_device_option(cam,option=rs_option.RS_OPTION_COLOR_SATURATION,value=70)
    pyrs.core.DeviceBase.set_device_option(cam,option=rs_option.RS_OPTION_COLOR_SHARPNESS,value=100)
    pyrs.core.DeviceBase.set_device_option(cam,option=rs_option.RS_OPTION_COLOR_WHITE_BALANCE,value=4600)
#---------------------------------------------------------------------------

def imgRead(cFileName, cadFileName, dFileName):
    global MaxArea,MinArea

    ImgOriginal = cv2.imread(cFileName)
    ImgOriginal = cv2.resize(ImgOriginal, ImageSize)

    ImgCad = cv2.imread(cadFileName)
    ImgCad = cv2.resize(ImgCad, ImageSize)
    ImgDepth = cv2.imread(dFileName)
    ImgDepth = cv2.resize(ImgDepth, ImageSize)

    ImgDepth = ImgDepth[65:415,65:490].copy()
    ImgDepth = cv2.resize(ImgDepth, ImageSize)
    ImgCroped = ImgCad[65:415,65:490].copy()
    ImgCad=cv2.resize(ImgCroped, ImageSize)

    ImgGrayBlured = cv2.cvtColor(ImgCad, cv2.COLOR_BGR2GRAY)
    Xc, Yc = ImageCenter
    MM=10
    Points= np.array([ImgDepth[Xc-MM,Yc-MM], ImgDepth[Xc+MM,Yc-MM],ImgDepth[Xc-MM,Yc+MM],ImgDepth[Xc+MM,Yc+MM]])
    Tmp=myRobot.GetDepthCenterCam(Points)
    # ------------------------------------------------
   # if (Tmp == 0):
   #     Tmp = myRobot.GetDepthCenterCam(Points)
    #cv2.imshow('imgGrayBlured ', ImgGrayBlured)
    #cv2.imshow('imgDepth ', ImgDepth)
    #cv2.imshow('ShapeMask ', ImgCad)
    #cv2.imshow('imgOriginal ', ImgOriginal)
   # key = cv2.waitKey(0)
   # if (key == 27):
    #      exit(0)


    Dthreshold = np.int16(Tmp * 1.4)

    MinArea = 500
    MaxArea = 50000
    if(Tmp <40) :
        MaxArea = 100000
        MinArea = 2000


   # Res = blend(ImgCad,ImgDepth, ImageSize)
    ImgDepth = cv2.cvtColor(ImgDepth, cv2.COLOR_BGR2GRAY)
    rec, ImgDepth = cv2.threshold(ImgDepth, Dthreshold, 255, cv2.THRESH_TOZERO_INV)
    Rate = 255 / Dthreshold
    ImgDepth = ImgDepth * Rate
    # --------------------------------
    kernel = np.ones((15, 15), np.uint8)
    ImgDepth = cv2.morphologyEx(ImgDepth, cv2.MORPH_ERODE, kernel, iterations=1)
    ImgDepth = cv2.morphologyEx(ImgDepth, cv2.MORPH_DILATE, kernel, iterations=2)
    rec, ImgDepth = cv2.threshold(ImgDepth, Dthreshold, 255, cv2.THRESH_BINARY)

    # --------------------------------
    kernel = np.ones((15, 15), np.uint8)
    ImgDepth = cv2.morphologyEx(ImgDepth, cv2.MORPH_DILATE, kernel, iterations=2)
    rec, ImgDepth = cv2.threshold(ImgDepth, 80, 255, cv2.THRESH_BINARY)
   # cv2.imshow("Depth", ImgDepth)
    kernel = np.ones((9, 9), np.uint8)

   # ImgGrayBlured = cv2.GaussianBlur(ImgGrayBlured, (15, 15), 0)
    blur5 = cv2.GaussianBlur(ImgGrayBlured, (11, 11), 0)
    blur3 = cv2.GaussianBlur(ImgGrayBlured, (17, 17), 0)
    ImgGrayBlured = (blur3 - blur5) * 10
    ImgGrayBlured = cv2.morphologyEx(ImgGrayBlured, cv2.MORPH_DILATE, kernel, iterations=1)
    ImgGrayBlured = cv2.GaussianBlur(ImgGrayBlured, (11, 11), 0)
    ImgGrayBlured = cv2.morphologyEx(ImgGrayBlured, cv2.MORPH_ERODE, kernel, iterations=1)


    return(ImgGrayBlured, ImgDepth, ImgOriginal)
#---------------------------------------------------------------------------
def myFilter (img):
    img = cv2.filter2D(img, 1, kernel1)
    return (img)
#---------------------------------------------------------------------------

def  SurfGen(img):
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kp, desc = surf.detectAndCompute(img, None)
    return(kp, desc)
#---------------------------------------------------------------------------
def MedianFilter(Points):
    P=Points.copy()
    M = np.size(P)
    P = (P.reshape(M))
    P.sort()
    L = P.__len__()
    if (L<1):
        return 0
    L = np.int16(L/2)
    return P[L]
#---------------------------------------------------------------------------
def  DepthCenter():
    global cam
    global DepthM
    global DDepths
    global ImgDepth
    global depth100 , cad100 , dac100 , depthscale100

    #dd = cam.depth * cam.depth_scale * DepthScale

    dd = np.uint8(dac100 * depthscale100 * 125)
    dd2 = cv2.resize(dd, ImageSize)
    # print  dd.shape, dd2.shape, ImageSize

    for i in range(0, DepthSamples):
        DDepths[:,:,i]= DDepths[:,:,i+1]
    DDepths[:, :, DepthSamples]=dd2

    DepthM  = np.median(DDepths, axis=2)

    ImgDepth = cv2.applyColorMap(DepthM.astype(np.uint8), cv2.COLORMAP_BONE)
    X11 = MedianFilter(DepthM[P11[0] - 2:P11[0] + 2, P11[1] - 2:P11[1] + 2])
    X13 = MedianFilter(DepthM[P13[0] - 2:P13[0] + 2, P13[1] - 2:P13[1] + 2])
    X22 = MedianFilter(DepthM[P22[0] - 2:P22[0] + 2, P22[1] - 2:P22[1] + 2])
    X31 = MedianFilter(DepthM[P31[0] - 2:P31[0] + 2, P31[1] - 2:P31[1] + 2])
    X33 = MedianFilter(DepthM[P33[0] - 2:P33[0] + 2, P33[1] - 2:P33[1] + 2])

    cv2.putText(ImgDepth, np.str(X11)[0:6], (P11[1], P11[0]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
    cv2.putText(ImgDepth, np.str(X13)[0:6], (P13[1], P13[0]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
    cv2.putText(ImgDepth, np.str(X22)[0:6], (P22[1], P22[0]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
    cv2.putText(ImgDepth, np.str(X31)[0:6], (P31[1], P31[0]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
    cv2.putText(ImgDepth, np.str(X33)[0:6], (P33[1], P33[0]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
    Points=np.array([X11,X13,X22,X31,X33])

    return (Points)

#---------------------------------------------------------------------------

def myFindContourMask(imgGrayBlured,Lw,Up):
    global shapeMask
    global shapeMask

    lower = np.array(Lw)
    upper = np.array(Up)
   #--------------------------------
    shapeMask = cv2.inRange(imgGrayBlured, lower, upper)

   # shapeMask = cv2.morphologyEx(shapeMask,cv2.MORPH_OPEN,kernel, iterations = 1)
   # shapeMask = cv2.morphologyEx(shapeMask, cv2.MORPH_DILATE, kernel, iterations=1)



    #---------------------------------------------------------
    (cnts, _) = cv2.findContours(shapeMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

   # print "I found %d black shapes" % (len(cnts))
    BigSizeCnts=[]
    h, w = shapeMask.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    mask[:] |= 0
    flags = 4
    flags |= cv2.FLOODFILL_FIXED_RANGE

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if ((area >= MaxArea) | (area <= MinArea)):
            P =  (cnt[0,0,0], cnt[0,0,1]) #(np.int16(ellipse[0][0]), np.int16(ellipse[0][1]))
            cv2.floodFill(shapeMask, mask , P, 0,flags=flags )
        else :
            x, y, w, h = cv2.boundingRect(cnt)
            if ((w>400) | (h>400)):
                P = (cnt[0, 0, 0], cnt[0, 0, 1])  # (np.int16(ellipse[0][0]), np.int16(ellipse[0][1]))
                cv2.floodFill(shapeMask, mask, P, 0, flags=flags)

    return (shapeMask)

#---------------------------------------------------------------------------

def myFindContour(shapeMask, imgOriginal):
