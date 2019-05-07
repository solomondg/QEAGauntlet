from neatolibs.pathlibs.mathutil.Pose2d import Pose2d
from neatolibs.pathlibs.mathutil.Twist2d import Twist2d
from neatolibs.pathlibs.mathutil.Translation2d import Translation2d
from neatolibs.pathlibs.mathutil.Rotation2d import Rotation2d
import math
import matplotlib.pyplot as plt
import numpy as np

# load target trajectory and robot encoders log
with open('trajectory.csv', 'r') as f:
    traj = np.asarray([[float(i) for i in j.split(',')] for j in f.read().split('\n')[:-1]])

with open('robot_pos_log.csv', 'r') as f:
    log = np.asarray([[float(i) for i in j.split(',')] for j in f.read().split('\n')[:-1]])

lV = np.gradient(log[:,1], log[:,0])
rV = np.gradient(log[:,2], log[:,0])

plt.plot(log[:,0], lV)
plt.plot(log[:,0], rV)

plt.plot(traj[:,0], traj[:,-2])
plt.plot(traj[:,0], traj[:,-1])

plt.show()

# copy and pasted from generateTrajectory.py output
p0 = np.asarray([3.2,-1.2])
t0 = np.asarray([0.009701039559043778,0.00242689749966174])

simRobot_pose = Pose2d(
    Translation2d(*p0),Rotation2d(*t0)
)

distAccum = 0

simx = []
simy = []
# forward kinematics from recorded encoder data
for i in range(len(log[:,0])-1):
    dl = log[i+1,1]-log[i,1]
    dr = log[i+1,2]-log[i,2]
    w = (dr-dl)/0.247613
    v = (dl+dr)/2
    distAccum += v
    simRobot_pose = simRobot_pose.relativeTransformBy(
        Pose2d.exp(Twist2d(dx=v,dy=0,dtheta=w))
    )
    simx.append(simRobot_pose.translation.x)
    simy.append(simRobot_pose.translation.y)

trajx = []
trajy = []
# forward kinematics from ideal trajecotry
simRobot_pose = Pose2d(
    Translation2d(*p0),Rotation2d(*t0)
)

for i in range(len(traj[:,0])-1):
    dt = traj[i+1,0]-traj[i,0]
    dl = traj[i,-2]*dt
    dr = traj[i,-1]*dt
    w = (dr-dl)/0.247613
    v = (dl+dr)/2
    simRobot_pose = simRobot_pose.relativeTransformBy(
        Pose2d.exp(Twist2d(dx=v,dy=0,dtheta=w))
    )
    trajx.append(simRobot_pose.translation.x)
    trajy.append(simRobot_pose.translation.y)

with open('points.csv', 'r') as f:
    pts = np.asarray(
        [[float(x) for x in i.split(',')] for i in f.read().split('\n')[:-1]]
    )

pts[:,0] = pts[:,0]*-0.005*0.6916+10+1
pts[:,1] = pts[:,1]*0.005*0.6916+-5.5+0.5

xBounds = [pts[:, 0].min(), pts[:, 0].max()]
yBounds = [pts[:, 1].min(), pts[:, 1].max()]

# compare them
plt.scatter(pts[:,0], pts[:,1])
plt.plot(trajx, trajy)
plt.plot(simx,simy)
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("LIDAR plot of arena, with desired and actual robot trajectory")
plt.legend(["Desired trajectory", "Robot trajectory"],loc=1)
plt.show()

X = np.asarray([0.        , 0.        , 0.        , 0.        , 1.15641262, 3.2       ])
Y = np.asarray([-4.03164564,  8.84813601, -5.94611395,  1.49087548,  0.16810969, -1.19825289])

# error functions for nonlinear least squares
getPt=lambda t:np.asarray([X@[t**i for i in range(6)][::-1],Y@[t**i for i in range(6)][::-1]])
getErr = lambda t, p: np.linalg.norm(getPt(t[0])-p)

from scipy.optimize import minimize

# get average path-path error
err = sum([minimize(getErr, 0.0, args=([trajx[i],trajy[i]]),bounds=[(0,1)],method='L-BFGS-B').fun for i in range(len(trajx))])/(sum([np.linalg.norm([trajx[i],trajy[i]]-np.asarray([trajx[i+1],trajy[i+1]])) for i in range(len(trajx)-1)])*len(trajx))
print("Average error: ",err)
print("Total distance: ",distAccum)
print("Time took: ",traj[-1,0])
