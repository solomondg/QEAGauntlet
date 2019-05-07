# gauntlet level 1


RANSACPlot.pdf contains ransac code


generateTrajectory generates potential field, performs gradient descent, and outputs path trajectory

follow openloop streams the aforementioned trajectory to the robot and records encoder data

plotResult compares the path described by the encoder data with the ideal path, and calculates total error, distance traveled, and time elapsed. 
