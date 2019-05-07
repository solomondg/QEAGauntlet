#!/usr/bin/env python2

import rospy
from std_msgs.msg import Float32MultiArray
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import scipy.interpolate
from time import time

# t,v,w,x,y,theta,lv,rv
# 0,1,2,3,4,5,    6, 7
with open('traj.csv', 'r') as f:
    csv = np.asarray([[float(i) for i in j.split(',')] for j in f.read().split('\n')[:-1]])

# comment out lines 19,20 to go backwards
lInterp = sp.interpolate.interp1d(csv[:,0],-csv[:,6][::-1])
rInterp = sp.interpolate.interp1d(csv[:,0],-csv[:,7][::-1])
lInterp = sp.interpolate.interp1d(csv[:,0],csv[:,6])
rInterp = sp.interpolate.interp1d(csv[:,0],csv[:,7])

encHist = []
offsets = []
startTime = None

# encoder recording
def callback(data):
    if startTime is not None:
        if len(offsets) == 0:
            offsets.extend(
                [data.data[0], data.data[1]]
            )
        encHist.append(
            [time()-startTime, data.data[0]-offsets[0], data.data[1]-offsets[1]]
        )

# boilerplate szn
pub = rospy.Publisher('raw_vel', Float32MultiArray, queue_size=10)
rospy.init_node("pathfollow")
encoders = rospy.Subscriber('encoders', Float32MultiArray, callback)

# i almost drove a neato off a table before i added this
raw_input("Press enter to start")
startTime = time()
r = rospy.Rate(20)

# nyoom
while time()-startTime < csv[-1,0] and not rospy.is_shutdown():
    pub.publish(
        Float32MultiArray(
            data=[
                float(lInterp(min(csv[-1,0],time()-startTime))),
                float(rInterp(min(csv[-1,0],time()-startTime)))
            ]
        )
    )

# stop robot
pub.publish(Float32MultiArray(data=[0,0]))
encoders.unregister()

# yeet data to disk
csv = ""
for i in encHist:
    csv += str(i[0]) + ',' + str(i[1]) + ',' + str(i[2]) + '\n'

with open('robot_pos_log.csv', 'w') as f:
    f.write(csv)
