import numpy as np
import scipy
import scipy.integrate

import matplotlib.pyplot as plt

from functionsPsiRho import psiFunc, psiSqFunc, trRho1Red, trRho1RedSq, rho1Red
from functionsPsiRho import GroundStateWavefunction

if __name__ == '__main__':
    a = 0.0
    b = 1.0

    if a == 0.0:
        xScale = np.sqrt(2.0/3.0*b) 
    else:
        xScale = np.sqrt(2.0*abs(b)/3.0/a)
        
    print("a = {0}, b = {1}".format(a, b))

    integral = scipy.integrate.nquad(
        psiSqFunc,
        ((-2 * xScale, +2 * xScale),
         (-2 * xScale, +2 * xScale)),
        args=(a, b))

    print("integral = {0}".format(integral))
    integral = scipy.integrate.nquad(
        psiSqFunc,
        ((-3 * xScale, +3 * xScale),
         (-3 * xScale, +3 * xScale)),
        args=(a, b))

    print("integral = {0}".format(integral))
    integral = scipy.integrate.nquad(
        psiSqFunc,
        ((-4 * xScale, +4 * xScale),
         (-4 * xScale, +4 * xScale)),
        args=(a, b))

    print("integral = {0}".format(integral))

    print("rho1Element[0, 0] = {0}".format(rho1Red(0, 0, a, b)))
    print("tr(rho1) {0}".format(trRho1Red(a, b)))
    print("tr(rho1^2) {0}".format(trRho1RedSq(a, b)))

