import math
import numpy as np

from neatolibs.pathlibs.mathutil.Types import Waypoint, Point2D, Vector2D
from pathgen.ParametricPolynomial import ParametricPolynomial


def boundRadians(angle: float) -> float:
    newAngle = math.fmod(angle, math.tau)
    if newAngle < 0:
        newAngle = math.tau + newAngle
    return newAngle


class ParametricSpline:
    def __init__(self, waypoints: [Waypoint]):
        self.splines = []
        for i in range(len(waypoints) - 1):
            splineStart: Waypoint = waypoints[i]
            splineEnd: Waypoint = waypoints[i + 1]

            # Position
            # fx = splineStart.x
            # fy = splineStart.y
            # ax + bx + cx + dx + ex + fx = splineEnd.x
            # ay + by + cy + dy + ey + fy = splineEnd.y

            # Heading
            # e = splineStart.t.x
            # e = splineStart.t.y
            # 5ax + 4bx + 3cx + 2dx + ex = splineEnd.t.x
            # 5ay + 4by + 3cy + 2dy + ey = splineEnd.t.y

            # Accel
            # 2dx = 0
            # 2dy = 0
            # 20ax + 12bx + 6cx + 2dx = 0
            # 20ay + 12by + 6cy + 2dy = 0

            # God fucking damnit I give up, hermite time
            # Thanks, Jaci...

            # xOffset = splineStart.x
            # yOffset = splineStart.y
            # delta = math.sqrt((splineEnd.x - splineStart.x) ** 2 + (splineEnd.y - splineStart.y) ** 2)
            # knotDistance = delta
            # angleOffset = math.atan2(splineEnd.y - splineStart.y, splineEnd.x - splineStart.x)
            # a0Delta = math.tan(boundRadians(splineStart.theta - angleOffset))
            # a1Delta = math.tan(boundRadians(splineEnd.theta - angleOffset))
            # d = knotDistance
            # sA = -(3 * (a0Delta + a1Delta)) / d ** 4
            # sB = (8 * a0Delta + 7 * a1Delta) / d ** 3
            # sC = -(6 * a0Delta + 4 * a1Delta) / d ** 2
            # sD = 0
            # sE = a0Delta

            # p(t) = h00->(2t^3 - 3t^2 + 1)p0 + h10->(t^3 - 2t^2 + t)m0 + h01->(-2t^3 + 3t^2)p1 + h11->(t^3-t^2)m1
            p0 = Point2D(splineStart.x, splineStart.y)
            p1 = Point2D(splineEnd.x, splineEnd.y)
            m0 = Vector2D(splineStart.cos, splineStart.sin)
            m1 = Vector2D(splineEnd.cos, splineEnd.sin)

            #      5  4  3   2  1  0
            h00 = [0, 0, 2, -3, 0, 1]
            h10 = [0, 0, 1, -2, 1, 0]
            h01 = [0, 0, -2, 3, 0, 0]
            h11 = [0, 0, 1, -1, 0, 0]
            h00, h10, h01, h11 = [np.asarray(i) for i in [h00, h10, h01, h11]]

            X = h00 * p0.x + h10 * m0.x + h01 * p1.x + h11 * m1.x
            Y = h00 * p0.y + h10 * m0.y + h01 * p1.y + h11 * m1.y
            # self.splines.append(ParametricPolynomial(x=list(X), y=list(Y)))

            # fuck that ^

            X, Y = [], []

            # Position
            # fx = splineStart.x
            # fy = splineStart.y
            # cx + dx + ex + fx = splineEnd.x
            # cy + dy + ey + fy = splineEnd.y

            # Heading
            # e = splineStart.t.x
            # e = splineStart.t.y
            # 3cx + 2dx + ex = splineEnd.t.x
            # 3cy + 2dy + ey = splineEnd.t.y

            coeffs = np.asarray([
                # 3  2  1  0  3  2  1  0
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                # [0, 3, 2, 1, 0, 0, 0, 0],
                # [0, 0, 0, 0, 0, 3, 2, 1]
                [3, 2, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 3, 2, 1, 0]
            ])

            # Goal vector:
            goal = np.asarray(
                [
                    splineStart.x,
                    splineStart.y,
                    splineEnd.x,
                    splineEnd.y,
                    splineStart.cos,
                    splineStart.sin,
                    splineEnd.cos,
                    splineEnd.sin
                ]
            )
            solved = np.linalg.solve(coeffs, goal)
            t5x, t4x, t3x, t2x, t1x, t0x = 0, 0, solved[0], solved[1], solved[2], solved[3]
            t5y, t4y, t3y, t2y, t1y, t0y = 0, 0, solved[4], solved[5], solved[6], solved[7]
            self.splines.append(ParametricPolynomial(x=[t5x, t4x, t3x, t2x, t1x, t0x], y=[t5y, t4y, t3y, t2y, t1y, t0y]))


if __name__ == "__main__":
    waypoints = [
        Waypoint(0, 0, 0),
        Waypoint(1, 0, 1),
        Waypoint(2, 1, 0.78),
        Waypoint(3, 2, 0.78),
        Waypoint(6, 3, -1.4),
        Waypoint(0, -2, -3.1),
        Waypoint(0, 0, 0)
    ]

    splines = ParametricSpline(waypoints=waypoints).splines

    for i in splines:
        print(i)