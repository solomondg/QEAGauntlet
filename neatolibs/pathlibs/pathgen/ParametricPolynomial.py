import time
from typing import List, Tuple, Any, Union

import numpy as np
from numpy.core.multiarray import ndarray
from numpy import roots
from scipy.optimize import least_squares
from scipy.optimize.nonlin import broyden1

from neatolibs.pathlibs.mathutil.Types import Point2D, toVec2D, Vector2D

epsilon = 1E-6

floatCompare = lambda x, y: abs(x - y) < epsilon


class ParametricPolynomial:
    X: ndarray # x(t) coefficient matrix
    Y: ndarray # y(t) coefficient matrix

    def __init__(self, *, x: List[float],
                 y: List[float]):
        assert len(x) == 6 and len(y) == 6
        self.X = np.asarray(x)
        self.Y = np.asarray(y)

    def getPoint(self, t: float) -> Point2D:
        assert 0 <= t <= 1
        t = np.asarray([t ** 5, t ** 4, t ** 3, t ** 2, t, 1])
        x = np.dot(self.X, t)
        y = np.dot(self.Y, t)
        return Point2D(x, y)

    def containsPoint(self, toTest: Point2D) -> List:
        # find roots of
        # <at^5 + bt^4 + ct^3 + dt^2 + et + f - toTest.x,
        # at^5 + bt^4 + ct^3 + dt^2 + et + f - toTest.y>
        # and return overlapping
        xRoots = [i for i in roots(np.asarray(self.X) - np.asarray([0, 0, 0, 0, 0, toTest.x])) if
                  i.imag == 0 and 0 <= i <= 1]
        yRoots = [i for i in roots(np.asarray(self.Y) - np.asarray([0, 0, 0, 0, 0, toTest.y])) if
                  i.imag == 0 and 0 <= i <= 1]
        same = []
        for i in xRoots:
            for j in yRoots:
                if abs(i.real - j.real) < epsilon:
                    same.append(i.real)

        for i in same:
            if not (floatCompare(toTest.x, self.getPoint(i).x) and floatCompare(toTest.y, self.getPoint(i).y)):
                return [False]

        return [True, *same]

    def project(self, point: Point2D, lastProjectionT: float = None) -> Tuple[Point2D, Any, Union[float, ndarray]]:
        assert type(point) == Point2D
        if lastProjectionT is not None:
            startGuess = lastProjectionT
        else:
            startGuess = 0.5

        def F(t: float):
            return np.linalg.norm((point - self.getPoint(t)).asList())

        ret = least_squares(F, startGuess, bounds=(0, 1), ftol=1E-4).x
        return self.getPoint(*ret), ret[0], np.linalg.norm((point - self.getPoint(*ret)).asList())

    def getDerivative(self):
        derivTransform = np.asarray([
            # at^5  + bt^4    + ct^3    + dt^2    + et    + f
            # 0*t^5 + 5*a*t^4 + 4*b*t^3 + 3*c*t^2 + 2*d*t + e

            # a  b  c  d  e  f
            [0, 0, 0, 0, 0, 0],  # a
            [5, 0, 0, 0, 0, 0],  # b
            [0, 4, 0, 0, 0, 0],  # c
            [0, 0, 3, 0, 0, 0],  # d
            [0, 0, 0, 2, 0, 0],  # e
            [0, 0, 0, 0, 1, 0]  # f
        ])
        # return QuinticPolynomial(x=list(np.matmul(self.X, derivTransform)), y=list(np.matmul(self.Y, derivTransform)))
        return ParametricPolynomial(x=list(np.matmul(derivTransform, self.X)),
                                 y=list(np.matmul(derivTransform, self.Y)))

    def getDerivativeAtPoint(self, t) -> Vector2D:
        return self.getDerivative().getPoint(t).asVector()

    def getSecondDerivative(self):
        return self.getDerivative().getDerivative()

    def getSecondDerivativeAtPoint(self, t) -> Vector2D:
        return self.getSecondDerivative().getPoint(t).asVector()

    def getUnitSecondDerivativeAtPoint(self, t) -> Vector2D:
        return self.getSecondDerivative().getPoint(t).asVector().normalize()

    def getTangentAtPoint(self, t) -> Vector2D:
        return self.getDerivativeAtPoint(t)

    def getUnitTangentAtPoint(self, t) -> Vector2D:
        tan = self.getTangentAtPoint(t).normalize()
        return tan

    def getNormalAtPoint(self, t) -> Vector2D:
        tan = self.getTangentAtPoint(t)
        return Vector2D(tan.y, -tan.x)

    def getUnitNormalAtPoint(self, t) -> Vector2D:
        tan = self.getUnitTangentAtPoint(t)
        return Vector2D(tan.y, -tan.x)

    def getSecondDerivativeNormalAtPoint(self, t) -> Vector2D:
        dNorm = self.getSecondDerivativeAtPoint(t)
        return Vector2D(dNorm.y, -dNorm.x)  # I think?  Lol

    def getUnitSecondDerivativeNormalAtPoint(self, t) -> Vector2D:
        dNorm = self.getSecondDerivativeAtPoint(t).normalize()
        return Vector2D(dNorm.y, -dNorm.x)  # I think?  Lol

    def getCurvatureAtPoint(self, t) -> float:
        return np.linalg.norm(self.getUnitSecondDerivativeAtPoint(t).asNDarray()) / \
               np.linalg.norm(self.getUnitTangentAtPoint(t).asNDarray())

    def getRadiusOfCurvatureAtPoint(self, t) -> float:
        return 1 / self.getCurvatureAtPoint(t)

    def __str__(self):
        x = list(self.X)
        y = list(self.Y)
        return \
            "({0}*t^5+{1}*t^4+{2}*t^3+{3}*t^2+{4}*t+{5},{6}*t^5+{7}*t^4+{8}*t^3+{9}*t^2+{10}*t+{11})".format(
                str(round(x[0], 5)),
                str(round(x[1], 5)),
                str(round(x[2], 5)),
                str(round(x[3], 5)),
                str(round(x[4], 5)),
                str(round(x[5], 5)),
                str(round(y[0], 5)),
                str(round(y[1], 5)),
                str(round(y[2], 5)),
                str(round(y[3], 5)),
                str(round(y[4], 5)),
                str(round(y[5], 5)))

        # return str(list(self.X)) + " " + str(list(self.Y))

    def __repr__(self):
        return self.__str__()