import numpy as np
import scipy
import scipy.special
import scipy.integrate
import logging

zeroApprox = 1e-10

################################################################################
def normbNeg00(a, b):

    factor = np.sqrt(2/np.pi)
    factorExp = np.sqrt(-b/a) * np.exp((b**2)/(4*a))
    besselTermP = scipy.special.iv(+0.25, (b**2)/(4*a))
    besselTermM = scipy.special.iv(-0.25, (b**2)/(4*a))

    norm = factor * (factorExp * (besselTermP + besselTermM))**(-0.5)
    return norm

def a4bNegFunc(a, b):

    factor = - (4 * a) / (np.pi**2 * b)
    expFactor = np.exp(b**2/(4*a))
    besselTermP = scipy.special.iv(+0.25, (b**2)/(4*a))
    besselTermM = scipy.special.iv(-0.25, (b**2)/(4*a))

    a4 = factor * (expFactor * (besselTermP + besselTermM))**(-2.0)
    return a4
################################################################################

def uFunc(x1, x1p, a, b):

    u = 4 * b + 3 * a * (x1**2 +  x1p**2)
    return u

def fFunc(x1, x1p, a, b):

    logging.debug("x1 = {0}, x1p = {1}, a = {2}, b = {3}".format(x1, x1p, a, b))

    u = uFunc(x1, x1p, a, b)
    u2_32a = u**2/32/a
    logging.debug("u = {0}, u2_32a = {1}".format(u, u2_32a))

    expFactor = np.exp(u2_32a)
    besselTerm = scipy.special.kv(0.25, u2_32a)
    logging.debug("expFactor = {0}, besselTerm = {1}".format(expFactor, besselTerm))
    
    f = np.sqrt(u/a) * expFactor * besselTerm 
    if np.isnan(f):
        f = np.sqrt(u/a) * scipy.special.kve(0.25, u2_32a)
    logging.debug("f = {0}".format(f))

    return f

def gFunc(x1, x1p, a, b):

    logging.debug("x1 = {0}, x1p = {1}, a = {2}, b = {3}".format(x1, x1p, a, b))
    u = uFunc(x1, x1p, a, b)
    u2_32a = u**2/32/a
    logging.debug("u = {0}, u2_32a = {1}".format(u, u2_32a))

    besselTermP = scipy.special.iv(+0.25, u2_32a)
    besselTermM = scipy.special.iv(-0.25, u2_32a)
    expFactor = np.exp(u2_32a)
    logging.debug(
            "expFactor = {0}, besselTermP = {1}, besselTermM = {2}".format(
                expFactor, besselTermP, besselTermM))

    g = np.pi * np.sqrt(-u/2/a) * expFactor * (besselTermP + besselTermM)
    logging.debug("g = {0}".format(g))

    return g

def fgCombinedFunc(x1, x1p, a, b):

    logging.debug("x1 = {0}, x1p = {1}, a = {2}, b = {3}".format(x1, x1p, a, b))
    u = uFunc(x1, x1p, a, b)
    u2_32a = u**2/32/a
    logging.debug("u = {0}, u2_32a = {1}".format(u, u2_32a))

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
    
    logging.debug("x1 = {0}, x1p = {1}, a = {2}, b = {3}, a4 = {4}".format(x1, x1p, a, b, a4))
    
    if abs(a4) <= zeroApprox:
        logA4 = np.log(a4)
        logExpFactor = -0.5 * b * (x1**2 + x1p**2) - 0.125 * a * (x1**4 + x1p**4)
        logPreFactor = logA4 + logExpFactor

        preFactor = 0.5 * np.exp(logPreFactor)
        logging.debug("preFactor = {0}".format(preFactor))
    else:
        expFactor =  np.exp(-0.5 * b * (x1**2 + x1p**2) - 0.125 * a * (x1**4 + x1p**4))
        preFactor = 0.5 * a4 * expFactor
        logging.debug("preFactor = {0}, expFactor={1}".format(preFactor, expFactor))

    return preFactor

def trPiece1(a, b, a4, infinityApprox):

    def integrandFunc(x1, a, b, a4):
        logging.debug("x1 = {0}, a = {1}, b = {2}, a4 = {3}".format(x1, a, b, a4))
        integrand = \
            rhoPreFacs(x1, x1, a, b, a4) * \
            fFunc(x1, x1, a, b)
        logging.debug("integrand = {0}".format(integrand))
        if np.isnan(integrand):
            logIntegrand = \
                    np.log(rhoPreFacs(x1, x1, a, b, a4)) + \
                    np.log(fFunc(x1, x1, a, b))
            integrand = np.exp(logIntegrand)
            logging.debug("Updated to integrand = {0}".format(integrand))

        return integrand

    integralNegEst, integralNegErr = \
        scipy.integrate.quad(
            integrandFunc,
            -infinityApprox, -np.sqrt(-2*b/(3*a)),
            args=(a, b, a4))
    
    integralPosEst, integralPosErr = \
        scipy.integrate.quad(
            integrandFunc,
            np.sqrt(-2*b/(3*a)), infinityApprox,
            args=(a, b, a4))

    trEst = integralNegEst + integralPosEst
    trErr = np.sqrt(integralNegErr**2 + integralPosErr**2)

    return trEst, trErr
    
def trPiece2(a, b, a4):

    def integrandFunc(x1, a, b, a4):
        logging.debug("x1 = {0}, a = {1}, b = {2}, a4 = {3}".format(x1, a, b, a4))
        integrand = \
            rhoPreFacs(x1, x1, a, b, a4) * \
            gFunc(x1, x1, a, b)
        logging.debug("integrand = {0}".format(integrand))
        if np.isnan(integrand):
            logIntegrand = \
                    np.log(rhoPreFacs(x1, x1, a, b, a4)) + \
                    np.log(gFunc(x1, x1, a, b))
            integrand = np.exp(logIntegrand)
            logging.debug("Updated to integrand = {0}".format(integrand))

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
def purityPiece1(a, b, a4, infinityApprox):

    def integrandFunc(x1, x1p, a, b, a4):

        integrand = \
            rhoPreFacs(x1, x1p, a, b, a4)**2 * \
            fFunc(x1, x1p, a, b)**2
        
        return integrand
    
    def upperBoundFunc(x1p, a, b):

        bound = -np.sqrt(-4/3 * b/a - x1p**2)
        return bound

    def lowerBoundFunc(x1p, a, b):

        bound = np.sqrt(-4/3 * b/a - x1p**2)
        return bound

    xBound = np.sqrt(-4/3*b/a)
    
    purityNegEst, purityNegErr = \
        scipy.integrate.dblquad(
        integrandFunc,
        -infinityApprox, -xBound,
        -infinityApprox, upperBoundFunc,
        args=(a, b, a4))

    purityPosEst, purityPosErr = \
        scipy.integrate.dblquad(
        integrandFunc,
        xBound, infinityApprox,
        lowerBoundFunc, infinityApprox,
        args=(a, b, a4))

    purityEst = purityPosEst + purityNegEst
    purityErr = np.sqrt(purityPosErr**2 + purityNegErr**2)

    return purityEst, purityErr

def purityPiece2(a, b, a4, infinityApprox):

    def integrandFunc(x1p, a, b, a4):

        integrand = \
            rhoPreFacs(x1, x1p, a, b, a4)**2 * \
            gFunc(x1, x1p, a, b)**2
        
        return integrand

    def upperBoundFunc(x1p, a, b):

        bound = np.sqrt(-4/3 * b/a - x1p**2)
        return bound

    def lowerBoundFunc(x1p, a, b):

        bound = -np.sqrt(-4/3 * b/a - x1p**2)
        return bound

    xBound = np.sqrt(-4/3*b/a)
    
    purityEst, purityErr = \
        scipy.integrate.dblquad(
        integrandFunc,
        -xBound, +xBound,
        lowerBoundFunc, upperBoundFunc,
        args=(a, b, a4))

    return purityEst, purityErr
################################################################################
