import numpy as np
import scipy
import scipy.special
import scipy.integrate

################################################################################
def psiFunc(x1, x2, a, b):

    exponent = \
        -0.5 * b * (x1**2 + x2**2) \
        - 0.125 * a * (x1**4 + x2**4 + 6 * x1**2 * x2**2)

    psi = np.exp(exponent)

    return psi

def psiSqFunc(x1, x2, a, b):

    psi = psiFunc(x1, x2, a, b)
    psiSq = psi * psi

    return psiSq

def rhoFunc(x1, x2, x1p, x2p, a, b):

    psi = psiFunc(x1, x2, a, b)
    psiP = psiFunc(x1p, x2p, a, b)

    rho = psi * psiP

    return rho
################################################################################
class GroundStateWavefunction:
    """
    Represents a ground state wavefunction for some potentials

    Parameters
    ----------
    a: float
    Coefficient a

    b: float
    Coefficient b
    """

    def __init__(self, a, b):
        self.a = a
        self.b = b

        self._find_norm_()

    def _find_norm_(self):
        """Numerically evaluates the norm"""

        a = self.a
        b = self.b

        if a == 0 and b != 0:
            xBound = 4 * np.sqrt(b)
        elif a != 0 and b == 0:
            xBound = 4 * np.sqrt(b)
        elif a != 0 and b != 0:
            if abs(b) < abs(a):
                xBound = 4*np.sqrt(3/2*abs(a/b))
            else:
                xBound = 4*np.sqrt(2/3*abs(b/a))
        else:
            raise ValueError("Cannot numerically handle a = %g, b = %g.")

        self._xBound_ = xBound

        self._integralEst_, self._integralErr_ = \
            scipy.integrate.nquad(
                psiSqFunc,
                ((-2 * xBound, +2 * xBound),
                 (-2 * xBound, +2 * xBound)),
                args=(a, b))

        self._normalizationFactor_ = 1/np.sqrt(self._integralEst_)


    def __call__(self, x1, x2):
        """
        Returns the value of the wavefunction evaluated at
        coordinates specified by the parameters

        Parameters
        ---------
        x1: float
        Position coordinate of the first oscillator

        x2: float
        Position coordinate of the second oscillator

        x1p: float
        Primed position coordinate of the first oscillator

        x2p: float
        Primed position coordinate of the second oscillator
        """
        
        psiUnnormalized = psiFunc(x1, x2, self.a, self.b)

        psiNormalized = psiUnnormalized * self._normalizationFactor_

        return psiNormalized
################################################################################
################################################################################
class WavefunctionDensityOperator:
    """
    Represents a density operator for a pure state

    Parameters
    ----------
    wavefunction: Wavefunction
    Object representing the wavefunction from which the density operator
    is to be evaluated
    """

    def __init__(self, wavefunction):
        self.wavefunction = wavefunction
        
    def element(self, x1, x2, x1p, x2p):
        """
        Returns the element of the density operator evaluated at
        coordinates specified by the parameters

        Parameters
        ---------
        x1: float
        Position coordinate of the first oscillator

        x2: float
        Position coordinate of the second oscillator

        x1p: float
        Primed position coordinate of the first oscillator

        x2p: float
        Primed position coordinate of the second oscillator
        """
        rhoElement = self.wavefunction(x1, x2) * self.wavefunction(x1p, x2p)

        return rhoElement

    def __call__(self, x1, x2, x1p, x2p):
        """
        Returns the element of the density operator evaluated at
        coordinates specified by the parameters

        Parameters
        ---------
        x1: float
        Position coordinate of the first oscillator

        x2: float
        Position coordinate of the second oscillator

        x1p: float
        Primed position coordinate of the first oscillator

        x2p: float
        Primed position coordinate of the second oscillator
        """
        return self.element(x1, x2, x1p, x2p)

################################################################################
################################################################################
class ReducedDensityOperator:
    """
    Represents a reduced density operator

    Parameters
    ----------
    fullDensityOperator: WavefunctionDensityOperator
    Object representing the full density operator from which the reduced
    density operator is to be calculated
    
    fullwavefunction: Wavefunction
    Object representing the wavefunction from which the reduced density
    operator is to be evaluated
    """

    def __init__(self, fullDensityOperator):
        self.fullDensityOperator = fullDensityOperator
        self.fullWavefunction = fullDensityOperator.wavefunction

    def element(self, x1, x1p):
        """
        Returns the element of the density operator evaluated at
        coordinates specified by the parameters

        Parameters
        ---------
        x1: float
        Position coordinate of the first oscillator

        x1p: float
        Primed position coordinate of the first oscillator
        """

        def integrandFunc(x2, x1, x1p):
            """
            Evaluates a diagonal element of the squared reduced density
            operator along coordinates of the second oscillator
            
            Parameters
            ----------
            x2: float
            Position coordinate of the second oscillator

            x1: float
            Position coordinate of the first oscillator

            x1p: float
            Primed position coordinate of the first oscillator
            """

            integrand = self.fullDensityOperator(x1, x2, x1p, x2)
            return integrand

        rho1ElementEst, rho1ElementErr = \
            scipy.integrate.quad(
                integrandFunc,
                -self.fullWavefunction._xBound_, self.fullWavefunction._xBound_,
                args=(x1, x1p))

        return rho1ElementEst, rho1ElementErr

    def __call__(self, x1, x1p):
        """
        Returns the element of the density operator evaluated at
        coordinates specified by the parameters

        Parameters
        ---------
        x1: float
        Position coordinate of the first oscillator

        x1p: float
        Primed position coordinate of the first oscillator
        """
        return self(x1, x1p)

    def squaredElement(self, x1, x1p):
        """
        Returns the element of the density operator evaluated at
        coordinates specified by the parameters

        Parameters
        ---------
        x1: float
        Position coordinate of the first oscillator

        x1p: float
        Primed position coordinate of the first oscillator
        """

        def integrandFunc(x2, x2p, x1, x1p):
            integrand = \
                self.fullDensityOperator(x1, x2, x1p, x2p) * \
                self.fullDensityOperator(x1, x2p, x1p, x2)
            return integrand

        rho1SqElementEst, rho1SqElementErr = \
            scipy.integrate.dblquad(
                integrandFunc,
                -self.fullWavefunction._xBound_, self.fullWavefunction._xBound_,
                -self.fullWavefunction._xBound_, self.fullWavefunction._xBound_,
                args=(x1, x1p))

        return rho1SqElementEst, rho1SqElementErr
################################################################################
################################################################################
def rho1Red(x1, x1p, a, b):
    """
    Evaluates an element of the reduced density operator
    along coordinates of the first oscillator
    
    Parameters
    ----------
    x1: float
    Position coordinate of the first oscillator

    x1p: float
    Primed position coordinate of the first oscillator

    a: float
    Coefficient a

    b: float
    Coefficient b
    """

    psiObject = GroundStateWavefunction(a, b)

    def rho2Diagonal(x2, x1, x1p):
        """
        Evaluates a diagonal element of the squared density operator
        along coordinates of the second oscillator

        Parameters
        ----------
        x2: float
        Position coordinate of the second oscillator

        x1: float
        Position coordinate of the first oscillator

        x1p: float
        Primed position coordinate of the first oscillator
        """

        rhoElement = psiObject(x1, x2) * psiObject(x1p, x2)
        return rhoElement

    rho1ElementEst, rho1ElementErr = scipy.integrate.quad(
        rho2Diagonal, -psiObject._xBound_, psiObject._xBound_,
        args=(x1, x1p))

    return rho1ElementEst, rho1ElementErr


def trRho1Red(a, b, flagUseOld=False):
    """
    Evaluates an element of the reduced density operator
    along coordinates of the first oscillator
    
    Parameters
    ----------
    x1: float
    Position coordinate of the first oscillator

    x1p: float
    Primed position coordinate of the first oscillator

    a: float
    Coefficient a

    b: float
    Coefficient b
    """

    if flagUseOld:
        psiObject = GroundStateWavefunction(a, b)
    
        def rho1RedDiagonal(x1):
            """
            Evaluates a diagonal element of the reduced density operator
            along coordinates of the second oscillator
    
            Parameters
            ----------
            x1: float
            Position coordinate of the first oscillator
            """
    
            rhoElementEst, rhoElementErr = rho1Red(x1, x1, a, b)
    
            return rhoElementEst
    
        rho1TrEst, rho1TrErr = scipy.integrate.quad(
            rho1RedDiagonal, -psiObject._xBound_, psiObject._xBound_,
            )
    else:
        psiObject = GroundStateWavefunction(a, b)
        rhoObject = WavefunctionDensityOperator(psiObject)
        rho1RedObject = ReducedDensityOperator(rhoObject)

        def rho1RedDiagonal(x1):
            """
            Evaluates a diagonal element of the squared reduced density operator
            along coordinates of the second oscillator
    
            Parameters
            ----------
            x1: float
            Position coordinate of the first oscillator
            """
    
            rhoElementEst, rhoElementErr = rho1RedObject.element(x1, x1)
    
            return rhoElementEst
    
        rho1TrEst, rho1TrErr = scipy.integrate.quad(
            rho1RedDiagonal, -psiObject._xBound_, psiObject._xBound_,
            )

    return rho1TrEst, rho1TrErr

def trRho1RedSq(a, b):
    """
    Evaluates an element of the reduced density operator
    along coordinates of the first oscillator
    
    Parameters
    ----------
    x1: float
    Position coordinate of the first oscillator

    x1p: float
    Primed position coordinate of the first oscillator

    a: float
    Coefficient a

    b: float
    Coefficient b
    """

    psiObject = GroundStateWavefunction(a, b)
    rhoObject = WavefunctionDensityOperator(psiObject)
    rho1RedObject = ReducedDensityOperator(rhoObject)

    def rho1RedSqDiagonal(x1):
        """
        Evaluates a diagonal element of the squared reduced density operator
        along coordinates of the second oscillator

        Parameters
        ----------
        x1: float
        Position coordinate of the first oscillator
        """

        rhoElementEst, rhoElementErr = rho1RedObject.squaredElement(x1, x1)

        return rhoElementEst

    rho1TrEst, rho1TrErr = scipy.integrate.quad(
        rho1RedSqDiagonal, -psiObject._xBound_, psiObject._xBound_,
        )

    return rho1TrEst, rho1TrErr
