import numpy as np
import scipy
import scipy.integrate

import matplotlib.pyplot as plt

from functionsPsiRho import psiFunc, psiSqFunc, trRho1Red, rho1Red
from functionsPsiRho import GroundStateWaveFunction

if __name__ == '__main__':
    a = 1.0
    b = -1.0

    numPoints = 256

    psiObject = GroundStateWaveFunction(a, b)

    xBound = np.sqrt(2*abs(b)/3/a)
    xArray = np.linspace(-3.0, +3.0, numPoints) * xBound
        
    psiArray = np.empty_like(xArray)
    psiSqArray = np.empty_like(xArray)

    for ix, x in enumerate(xArray):
        psiArray[ix] = psiObject(x, x)
        psiSqArray[ix] = psiArray[ix] * psiArray[ix]
    
    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    ax.plot(xArray, psiSqArray)

    ax.set_ylabel(r'$\psi^2$')
    ax.set_xlabel(r'$x$')

    ax.axvline(-2*xBound, ls='dashed', lw=0.5, color='black', alpha=0.5)
    ax.axvline(-xBound, ls='dashed', lw=0.5, color='black', alpha=0.5)
    ax.axvline(-0.5*xBound, ls='dashed', lw=0.5, color='black', alpha=0.5)
    ax.axvline(-0.25*xBound, ls='dashed', lw=0.5, color='black', alpha=0.5)
    ax.axvline(0, ls='dashed', lw=0.5, color='black', alpha=0.5)
    ax.axvline(+0.25*xBound, ls='dashed', lw=0.5, color='black', alpha=0.5)
    ax.axvline(+0.5*xBound, ls='dashed', lw=0.5, color='black', alpha=0.5)
    ax.axvline(+xBound, ls='dashed', lw=0.5, color='black', alpha=0.5)
    ax.axvline(+2*xBound, ls='dashed', lw=0.5, color='black', alpha=0.5)

    plt.savefig("numerical_b={0}.pdf".format(b))
    plt.close()
