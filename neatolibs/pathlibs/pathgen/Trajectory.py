import math
from typing import List

from neatolibs.pathlibs.mathutil.Pose2d import Pose2d
from neatolibs.pathlibs.mathutil.Twist2d import Twist2d


class Trajectory:
    def __init__(self, dt, pose: List[Pose2d], vel: List[Twist2d], timeOffset=0.0):
        assert len(pose) == len(vel)
        self.dt = dt
        self.pose = pose
        self.vel = vel

        self.startTime = timeOffset
        self.totalTime = dt*len(pose)
        self.endTime = self.startTime + self.totalTime

    def getState(self, t):
        assert self.startTime <= t <= self.endTime
        low = int(math.floor((t-self.startTime)/self.dt))
        high = int(math.ceil((t-self.startTime)/self.dt))
        closest = int(round((t-self.startTime)/self.dt))
        if high > len(self.pose) and closest <= len(self.pose):
            return [self.pose[closest], self.vel[closest]]
        elif closest > len(self.pose):
            return [self.pose[low], self.vel[low]]

        assert low <= closest <= high <= len(self.pose)

        residual = high - (t-self.startTime)/self.dt
        pos_high = self.pose[high].translation
        rot_high = self.pose[high].rotation
        vel_high = self.vel[high]
        pos_low = self.pose[low].translation
        rot_low = self.pose[low].rotation
        vel_low = self.vel[low]

