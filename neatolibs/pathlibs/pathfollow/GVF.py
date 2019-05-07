import time

import numpy as np

from typing import Callable
from pathgen.ParametricPolynomial import ParametricPolynomial
from neatolibs.pathlibs.mathutil.Types import Point2D, Vector2D, toVec2D


class GVFResult:
    normalizedGradient: Vector2D
    curvature: Vector2D
    gradient: Vector2D
    error: float

    def __init__(self, gradient: Vector2D, error, normalizedGradient: Vector2D, curvature: Vector2D):
        self.normalizedGradient = normalizedGradient
        self.curvature = curvature
        self.gradient = gradient
        self.error = error


class SplineGVF:
    spline: ParametricPolynomial
    distTransform: Callable
    lastProjResult: float = None
    lastProjPoint: Point2D = None
    lastProjNorm: float = None
    Ks: float
    Kn: float

    def __init__(self, *,
                 spline: ParametricPolynomial,
                 Ks: float,  # Error scale constant
                 Kn: float,  # Tangent-normal tradeoff constant
                 dist_transform: Callable = lambda l: l ** 2):
        self.spline = spline
        self.distTransform = dist_transform
        self.Ks = Ks
        self.Kn = Kn

    def setSpline(self, spline):
        self.spline = spline

    def getError(self, point: Point2D) -> float:
        projPoint: Point2D
        t: float
        error: float
        projPoint, t, error = self.project(point)
        self.lastProjPoint, self.lastProjResult, self.lastProjNorm = projPoint, t, error
        self.lastProjResult = t
        return self.distTransform(error) * self.Ks

    def setSpline(self, spline: ParametricPolynomial):
        if spline != self.spline:
            self.spline = spline
            self.lastProjResult = None

    def project(self, point: Point2D):
        res0 = self.spline.project(point=point, lastProjectionT=0)
        res05 = self.spline.project(point=point, lastProjectionT=0.5)
        res1 = self.spline.project(point=point, lastProjectionT=1)
        res = res0
        if res05[2] < res[2]:
            res = res05
        if res1[2] < res[2]:
            res = res1
        return res

    def getVectorFieldAtPoint(self, point: Point2D) -> GVFResult:
        # overall eq is t(t) - Ke*e(t)*n(t)
        projPoint, t, n = self.project(point)
        self.lastProjPoint, self.lastProjResult, self.lastProjNorm = projPoint, t, n
        error = self.getError(point)  # fuck tha (optimization) police
        tan: Vector2D = self.spline.getTangentAtPoint(t)
        normal: Vector2D = self.spline.getNormalAtPoint(t)
        # print("tan", tan)
        # print("normal", normal)
        # print("error", error)
        # print("kn", self.Kn)
        # print("grad", grad)

        projToRobot = (point - projPoint).asVector().normalize()
        norm = normal.normalize()
        antiNorm = toVec2D(-(norm.asNDarray()))

        def clamp(low, num, high):
            return max(low, min(num, high))

        try:
            normAngle = np.math.acos(clamp(-1, np.dot(projToRobot.asNDarray(), norm.asNDarray()), 1))
            antiNormAngle = np.math.acos(clamp(-1, np.dot(projToRobot.asNDarray(), antiNorm.asNDarray()), 1))
        except Exception as e:
            print("exception")
            print(np.dot(projToRobot.asNDarray(), norm.asNDarray()))
            print(np.dot(projToRobot.asNDarray(), antiNorm.asNDarray()))
            raise e

        if normAngle < antiNormAngle:
            grad: Vector2D = toVec2D(tan.asNDarray() - self.Kn * error * normal.asNDarray())
        else:
            grad: Vector2D = toVec2D(tan.asNDarray() - self.Kn * error * -normal.asNDarray())

        # Oh lord, it's hessian time....
        # d/dt v(t) = t'(t) - Ke*e*n'(t)
        # If this doesn't work, maybe find discrete deriv of v(t+e*v(t)),v(t),v(t-e*v(t) ? That could work....
        # TODO I guess?
        dTan_dt = self.spline.getSecondDerivativeAtPoint(t)
        dNormal_dt = self.spline.getSecondDerivativeNormalAtPoint(t)
        hessian: Vector2D = toVec2D(dTan_dt.asNDArray() - self.Kn * error * dNormal_dt.asNDarray())

        return GVFResult(
            gradient=grad,
            error=error,
            normalizedGradient=grad.normalize(),
            curvature=hessian.normalize(),
        )


# We're going to test this with the polynomial
# x = [2 3 -1 1]
# y = [3 2 -4 1.5]

if __name__ == "__main__":
    testSpline = ParametricPolynomial(
        x=[0, 0, -0.4597, 0.4597, 1, 0],
        y=[0, 0, 0.84147, -0.84147, 0, 0]
    )

    GVF = SplineGVF(spline=testSpline, Ks=24, Kn=24.0) # kn=0.0 means 100% tangent, kn=inf means 100% normal

    xTestBounds = [-0.25, 1.25]
    yTestBounds = [-0.25, 0.25]
    testingRes = 0.050

    samples = ""

    xTest = list(
        np.linspace(xTestBounds[0], xTestBounds[1], num=int((xTestBounds[1] - xTestBounds[0]) / testingRes + 1)))
    yTest = list(
        np.linspace(yTestBounds[0], yTestBounds[1], num=int((yTestBounds[1] - yTestBounds[0]) / testingRes + 1)))

    last = 0.5
    start = time.time()
    inc = 0
    for x in xTest:
        for y in yTest:
            res = GVF.getVectorFieldAtPoint(Point2D(x, y))  # Projected point, t value, error
            normalizedGrad = res.normalizedGradient
            u = normalizedGrad.x
            v = normalizedGrad.y
            e = res.error
            samples += str(x) + "," + str(y) + "," + str(u) + "," + str(v) + "\n"
            #samples += str(x) + "," + str(y) + "," + str(e) + "\n"
            inc += 1
        print(x)

    total = time.time() - start

    print(GVF.project(Point2D(0.2, 0.05)))
    x, y, _ = GVF.project(Point2D(0.2, 0.05))
    t = GVF.spline.containsPoint(x)[1]
    print("p", GVF.spline.containsPoint(x))
    print("t", GVF.spline.getUnitTangentAtPoint(t))
    print("n", GVF.spline.getUnitNormalAtPoint(t))
    print("v", GVF.getVectorFieldAtPoint(Point2D(0.2, 0.05)).normalizedGradient)

    print("v2", toVec2D(
        GVF.spline.getTangentAtPoint(t).asNDarray() - 0 * 1.16 * GVF.spline.getNormalAtPoint(
            t).asNDarray()).normalize())

    with open('vectorField.csv', 'w') as f:
        f.write(samples)