import numpy as np
import scipy
import scipy.special
import scipy.integrate

################################################################################
def normbNeg00(a, b):

    factor = np.sqrt(2/np.pi)
    factorExp = np.sqrt(-b/a) * np.exp((b**2)/(4*a))
    besselTermP = scipy.special.iv(+0.25, (b**2)/(4*a))
    besselTermM = scipy.special.iv(-0.25, (b**2)/(4*a))

    norm = factor * (factorExp * (besselTermP + besselTermM))**(-0.5)

    return norm
################################################################################

def uFunc(x1, x1p, a, b):

    u = 4 * b + 3 * a * (x1**2 +  x1p**2)
    return u

def fFunc(x1, x1p, a, b):
    u = uFunc(x1, x1p, a, b)
    u2_32a = u**2/32/a
 
    expFactor = 1.0
    expFactor = np.exp(u2_32a)
    
    f = np.sqrt(u/a) * expFactor * scipy.special.kv(0.25, u2_32a)

    return f

def gFunc(x1, x1p, a, b):

    u = uFunc(x1, x1p, a, b)
    u2_32a = u**2/32/a

    besselTermP = scipy.special.iv(+0.25, u2_32a)
    besselTermM = scipy.special.iv(-0.25, u2_32a)

    expFactor = 1.0
    expFactor = np.exp(u2_32a)
    
    
    g = np.pi * np.sqrt(-u/2/a) * expFactor * (besselTermP + besselTermM)

    return g

def fgCombinedFunc(x1, x1p, a, b):
    u = uFunc(x1, x1p, a, b)
    u2_32a = u**2/32/a

    if u < 0:
        besselTermP = scipy.special.ive(+0.25, u2_32a)
        besselTermM = scipy.special.ive(-0.25, u2_32a)
    
        fg = np.pi * np.sqrt(-u/2/a) * (besselTermP + besselTermM)
        
    elif u > 0:
        fg = np.sqrt(u/a) * scipy.special.kve(0.25, u2_32a)

    return fg

################################################################################
################################################################################
def rhoPreFacs(x1, x1p, a, b, a4):

    pf = 0.5 * a4 * np.exp(-0.5 * b * (x1**2 + x1p**2) - 0.125 * a * (x1**4 + x1p**4))

    return pf

def trPiece1(a, b, a4, infinity_approx):

    def integrandFunc(x1, a, b, a4):
        integrand = \
            rhoPreFacs(x1, x1, a, b, a4) * \
            fFunc(x1, x1, a, b)
        
        return integrand

    integralNegEst, integralNegErr = \
        scipy.integrate.quad(
            integrandFunc,
            -infinity_approx, -np.sqrt(-2*b/(3*a)),
            args=(a, b, a4))
    
    integralPosEst, integralPosErr = \
        scipy.integrate.quad(
            integrandFunc,
            np.sqrt(-2*b/(3*a)), infinity_approx,
            args=(a, b, a4))

    trEst = integralNegEst + integralPosEst
    trErr = np.sqrt(integralNegErr**2 + integralPosErr**2)

    return trEst, trErr
    
def trPiece2(a, b, a4):

    def integrandFunc(x1, a, b, a4):
        integrand = \
            rhoPreFacs(x1, x1, a, b, a4) * \
            gFunc(x1, x1, a, b)

        return integrand

    integralEst, integralErr = \
        scipy.integrate.quad(
            integrandFunc,
            -np.sqrt(-2*b/(3*a)), +np.sqrt(-2*b/(3*a)),
            args=(a, b, a4))
    
    trEst = integralEst
    trErr = integralErr

    return trEst, trErr
################################################################################
################################################################################
def purityPiece1(a, b, a4, infinity_approx):

    def integrandFunc(x1, x1p, a, b):

        integrand = \
            rhoPreFacs(x1, x1p, a, b, a4)**2 * \
            fFunc(x1, x1p, a, b)**2
        
        return integrand

    xBound = np.sqrt(-2/3*b/a)
    
    purityNegEst, purityNegErr = \
        scipy.integrate.nquad(
        integrandFunc,
        ((-infinity_approx, -xBound), (-infinity_approx, xBound)),
        args=(a, b, a4))

    purityPosEst, purityPosErr = \
        scipy.integrate.nquad(
        integrandFunc,
        ((xBound, infinity_approx), (xBound, infinity_approx)),
        args=(a, b, a4))

    purityEst = purityPosEst + purityNegEst
    purityErr = np.sqrt(purityPosErr**2 + purityNegErr**2)

    return purityEst, purityErr

def purityPiece2(a, b, a4, infinity_approx):

    def integrandFunc(x1, x1p, a, b):

        integrand = \
            rhoPreFacs(x1, x1p, a, b, a4)**2 * \
            gFunc(x1, x1p, a, b)**2
        
        return integrand

    xBound = np.sqrt(-2/3*b/a)
    
    purityEst, purityErr = \
        scipy.integrate.nquad(
        integrandFunc,
        ((-xBound, +xBound), (-xBound, +xBound)),
        args=(a, b, a4))

    return purityEst, purityErr
################################################################################
