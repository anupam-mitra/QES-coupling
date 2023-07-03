import numpy as np
import matplotlib.pyplot as plt

import functions

if __name__ == '__main__':

    a = 1.0
    b = -1.0

    numPoints = 256

    # x
    xArray = np.linspace(-6.0, +6.0, numPoints)
    
    fgArray = np.empty_like(xArray)
    uArray = np.empty_like(xArray)
    
    for ix, x in enumerate(xArray):
        fgArray[ix] = functions.fgCombinedFunc(x, x, a, b)
        uArray[ix] = functions.uFunc(x, x, a, b)
        
    #fgArray[np.isnan(fgArray)] = -1

    print("xArray = %s" % (xArray,))
    print("fgArray = %s" % (fgArray,))
    
    fig, (axU, axU2, axFG) = plt.subplots(3, 1, constrained_layout=True)

    axU.plot(xArray, uArray)
    axU2.plot(xArray, uArray**2/32)
    axFG.plot(xArray, fgArray)

    axU.set_ylabel(r'$u(x)$')

    axU2.set_ylabel(r'$u^2(x)/32$')
    
    axFG.set_ylabel(r'$f(x), g(x)$')

    for ax in (axU, axU2, axFG):
        ax.grid()
        ax.set_xticks(range(-6, +7, 1))
        ax.set_xlabel(r'$x$')
    
    plt.show()
