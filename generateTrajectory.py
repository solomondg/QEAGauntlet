#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import scipy as sp
import scipy.interpolate
from multiprocessing import Pool
from neatolibs.pathlibs.mathutil.Pose2d import Pose2d
from neatolibs.pathlibs.mathutil.Twist2d import Twist2d
from neatolibs.pathlibs.mathutil.Translation2d import Translation2d
from neatolibs.pathlibs.mathutil.Rotation2d import Rotation2d

# load laser scan
with open('points.csv', 'r') as f:
    pts = np.asarray(
        [[float(x) for x in i.split(',')] for i in f.read().split('\n')[:-1]]
    )

# center, scale
pts[:,0] = pts[:,0]*-0.005*0.6916+10+1
pts[:,1] = pts[:,1]*0.005*0.6916+-5.5+0.5

#plot bounds
xBounds = [pts[:, 0].min(), pts[:, 0].max()]
yBounds = [pts[:, 1].min(), pts[:, 1].max()]

plt.scatter(pts[:,0], pts[:,1])

# x/y sampling density for potential field
axisPts = 250
#axisPts = 100

# position of sink
sinkPt = np.asarray([4.55, -0.5])

sigma = 0.35
scale = 1
bound = 1

# awful lambda abuse
sourceFcn = lambda d: scale/(np.sqrt(2.*np.pi*sigma**2)) * np.exp(-(d**2)/(2.*sigma**2))
sinkFcn = lambda d: -sourceFcn(d)
# robot start point
currentPt = np.asarray([3.2,-1.2])
maxZ = sourceFcn(0)
# more awful lambda abuse
fieldFcn = lambda x,y: \
    sourceFcn(np.linalg.norm(np.asarray([x,y])-[3.0,-1.25]))/maxZ + \
    sinkFcn(np.linalg.norm(np.asarray([x,y])-sinkPt))/maxZ



repelScale = 0.01
repelROI = 0.225
repelFcn = lambda d: \
    min(maxZ, repelScale/d**2) if d<=repelROI else 0

ptsDiv = 10
# awful unreadable code, just gets the potential field at the given point while
# ignoring bucket points
evalPt = lambda x,y: fieldFcn(x,y) + \
    (len([1 for i in pts[::ptsDiv] if np.linalg.norm([x,y]-i)<repelROI and not all([4.33<i[0]<4.8,-.65<i[1]<-.35])])*repelScale)**1.5

# concurrency szn
def evalPts(x):
    return [[x,y,evalPt(x,y)] for y in np.linspace(yBounds[0], yBounds[1], axisPts)]

# throw more cores at the problem (this samples the potential field at a bunch
# of points)
fieldPoints = []
with Pool(10) as p:
    # samples
    fieldPoints = np.concatenate(np.asarray(
        list(tqdm(p.imap(evalPts, np.linspace(xBounds[0], xBounds[1], axisPts)), total=axisPts))
    ))

fieldPoints = np.asarray(fieldPoints)

contourLevelsCount = 50
contourLevels = np.linspace(-maxZ, maxZ, contourLevelsCount)

# pretty pictures
plt.tricontour(fieldPoints[:,0], fieldPoints[:,1], fieldPoints[:,2],
               levels=contourLevels)
plt.scatter([currentPt[0], sinkPt[0]], [currentPt[1], sinkPt[1]])
plt.show()

# how far away from the sink we have to be to count as touching the bucket
sinkEndDist = 0.25

# get gradient at given point
def getGrad(x,y):
    dd = 0.01
    #z0 = fitFcn(x,y)
    dx = (evalPt(x+dd/2,y)-evalPt(x-dd/2,y))/dd
    dy = (evalPt(x,y+dd/2)-evalPt(x,y-dd/2))/dd
    #dx = (fitFcn(x+dd/2,y)-fitFcn(x-dd/2,y))/dd
    #dy = (fitFcn(x,y+dd/2)-fitFcn(x,y-dd/2))/dd
    return (np.asarray([dx,dy]) / np.linalg.norm(np.asarray([dx,dy]))).reshape((2,))


ptHistory = []
stepSize = 0.001
#stepSize = 0.01
i=0
# garbage gradient descent impl
while np.linalg.norm(currentPt-sinkPt) > sinkEndDist:
    i+=1
    print("Iteration {0}, XY: {1}, error: {2}".format(i, currentPt, np.linalg.norm(currentPt-sinkPt)))
    grad = getGrad(currentPt[0], currentPt[1])
    ptHistory.append(currentPt.copy())
    currentPt -= grad*stepSize

ptHistory = np.asarray(ptHistory)

# more pretty pictures
plt.tricontour(fieldPoints[:,0], fieldPoints[:,1], fieldPoints[:,2],
               levels=contourLevels)
plt.scatter([currentPt[0], sinkPt[0]], [currentPt[1], sinkPt[1]])
#plt.show()
plt.plot(ptHistory[:,0], ptHistory[:,1])
plt.scatter(pts[:,0], pts[:,1])

# write path
csv = ""
for i in ptHistory:
    csv += str(i[0]) + "," + str(i[1]) + "\n"

with open('path.csv', 'w') as f:
    f.write(csv)

# fit spline to points for easy angular velocity and path length
P = np.polyfit((ptHistory[:,0]-ptHistory[0,0])/(ptHistory[-1,0]-ptHistory[0,0]),ptHistory[:,1],deg=5)
# time for sympy abuse
from sympy import symbols, N, integrate, Function, Matrix
u_ = symbols('u', real=True)
eq = Function('t')
X = np.asarray([0, 0, 0, 0, ptHistory[-1,0]-ptHistory[0,0], ptHistory[0,0]])
Y = P
eqn = Matrix([
    X[0]*u_**5 + X[1]*u_**4 + X[2]*u_**3 + X[3]*u_**2 + X[4]*u_ + X[5],
    Y[0]*u_**5 + Y[1]*u_**4 + Y[2]*u_**3 + Y[3]*u_**2 + Y[4]*u_ + Y[5],
])
print(*ptHistory[0])
print(*ptHistory[1]-ptHistory[0])

plt.show()
# compares the spline with the gradient descent'd path
plt.plot(ptHistory[:,0], ptHistory[:,1])
plt.plot(
    [eqn.subs(u_,i)[0] for i in np.linspace(0,1,100)],
    [eqn.subs(u_,i)[1] for i in np.linspace(0,1,100)]
)
plt.legend(["Actual", "Spline"])
plt.show()
pathLength = N(integrate(eqn.diff(u_).norm(), (u_, 0, 1)))

neatoMaxVel = 0.3 # m/s
profileCruiseVel = neatoMaxVel*0.75 # m/s
profileAccel = 0.3 # m/s^2

# lots of awful recursive code for trapezoidal profile gen
cruiseVelStartTime = profileCruiseVel/profileAccel
dt = 0.02 # s
tVec = [0.0] # time (s)
xVec = [0.0] # distance traveled (m)
vVec = [0.0] # velocity (m/s)
aVec = [0.0] # acceleration (m/s^2)
cruiseVelTime = pathLength/profileCruiseVel - cruiseVelStartTime
if profileCruiseVel * cruiseVelStartTime > pathLength:
    cruiseVelStartTime = math.sqrt(pathLength/profileAccel)
    cruiseVelEndTime = cruiseVelStartTime
    timeTotal = 2.0 * cruiseVelStartTime
    profile_max_v = profileAccel*cruiseVelStartTime
else:
    cruiseVelEndTime = cruiseVelStartTime + cruiseVelTime
    timeTotal = cruiseVelEndTime + cruiseVelStartTime

while tVec[-1] < timeTotal:
    t = tVec[-1]+dt
    tVec.append(t)

    if t < cruiseVelStartTime:
        aVec.append(profileAccel)
        vVec.append(profileAccel * t)
    elif t < cruiseVelEndTime:
        aVec.append(0.0)
        vVec.append(profileCruiseVel)
    elif t < timeTotal:
        decelStartTime = t - cruiseVelEndTime
        aVec.append(-profileAccel)
        vVec.append(profileCruiseVel - profileAccel * decelStartTime)
    else:
        aVec.append(0.0)
        vVec.append(0.0)
    # print(xVec[-1], vVec[-1], dt, vVec[-1]*dt)
    xVec.append(xVec[-1] + vVec[-1] * dt)

# in case you want to look at the profile
#plt.plot(tVec, xVec)
#plt.plot(tVec, vVec)
#plt.plot(tVec, aVec)
#plt.xlabel("Time (s)")
#plt.legend(["Position (m)", "Velocity (m/s)", "Velocity (m/s^2)"])
#plt.show()

# determined empirically
wheelbase = 0.249

wVec = []

# yay, math
def getW(percent):
    tan = eqn.diff(u_) / eqn.diff(u_).norm()
    dtan = tan.diff(u_)
    tan_sub = tan.subs(u_, percent)
    dtan_sub = dtan.subs(u_, percent)
    return float(np.cross([float(i) for i in tan_sub], [float(i) for i in dtan_sub]))

# generates angular velocities for all path points
for i in tqdm(range(len(xVec))):
    percent = xVec[i] / pathLength
    w = getW(percent) * vVec[i]
    wVec.append(w)

w = np.asarray(wVec)
v = np.asarray(vVec)

traj_l = v - w * (wheelbase / 2)
traj_r = v + w * (wheelbase / 2)
#plt.plot(tVec, traj_l)
#plt.plot(tVec, traj_r)
#plt.legend(["left","right"])
#plt.show()

# verify path by running simulated robot off of trajectory
robotPose = Pose2d(
    Translation2d(*ptHistory[0]),
    Rotation2d(*(ptHistory[1]-ptHistory[0]))
)

simx = [robotPose.translation.x]
simy = [robotPose.translation.y]

for i in range(len(traj_r)):
    _w = (traj_r[i]-traj_l[i])/wheelbase
    _v = (traj_l[i]+traj_r[i])/2
    robotPose = robotPose.relativeTransformBy(robotPose.exp(
        Twist2d(dx=_v*dt, dy=0, dtheta=_w*dt)
    ))
    simx.append(robotPose.translation.x)
    simy.append(robotPose.translation.y)

# more pretty pictures
plt.tricontour(fieldPoints[:,0], fieldPoints[:,1], fieldPoints[:,2],
               levels=contourLevels)
plt.scatter([0, sinkPt[0]], [0, sinkPt[1]])

plt.plot(ptHistory[:,0], ptHistory[:,1])
plt.plot(simx, simy)
plt.show()

# t, v, w, x, y, theta, lV, rV
csv = ""

# dump everything to disk
for i in range(len(traj_r)):
    csv += \
        "{0},{1},{2},{3},{4},{5},{6},{7}\n".format(
            tVec[i],
            v[i],
            w[i],
            0,0,0,
            traj_l[i],
            traj_r[i]
        )

with open("trajectory.csv", 'w') as f:
    f.write(csv)
