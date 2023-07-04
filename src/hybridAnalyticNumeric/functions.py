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
    
    f = np.sqrt(abs(u/a)) * expFactor * besselTerm 
    if np.isnan(f):
        besselTerm = scipy.special.kve(0.25, u2_32a)
        logging.debug("besselTerm = {0}".format(besselTerm))
        f = np.sqrt(abs(u/a)) * besselTerm
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
    
    expFactor =  np.exp(-0.5 * b * (x1**2 + x1p**2) - 0.125 * a * (x1**4 + x1p**4))
    preFactor = 0.5 * a4 * expFactor
    logging.debug("preFactor = {0}, expFactor={1}".format(preFactor, expFactor))

    return preFactor

def trPiece1(a, b, a4, infinityApprox):

    def integrandFunc(x1, a, b, a4):
        if x1**2 < -2/3*b/a:
            integrand = 0.0
        else:
            integrand = \
                rhoPreFacs(x1, x1, a, b, a4) * \
                fFunc(x1, x1, a, b)
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

        if x1**2 > -2/3*b/a:
            integrand = 0.0
        else:
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
def purityPiece1(a, b, a4, infinityApprox):

    xBound = np.sqrt(-4/3*b/a)

    def integrandFunc(x1, x1p, a, b, a4):

        if (x1**2 + x1p**2) < xBound**2:
            integrand = 0.0
        else:
            integrand = \
                rhoPreFacs(x1, x1p, a, b, a4)**2 * \
                fFunc(x1, x1p, a, b)**2
        
        return integrand
    
    purityEst, purityErr = \
        scipy.integrate.dblquad(
        integrandFunc,
        -infinityApprox, infinityApprox,
        -infinityApprox, infinityApprox,
        args=(a, b, a4))

    logging.info("purity = {0} +- {1}".format(purityEst, purityErr))

    return purityEst, purityErr

def purityPiece2(a, b, a4):

    xBound = np.sqrt(-4/3*b/a)
        
    def integrandFunc(x1, x1p, a, b, a4):

        if (x1**2 + x1p**2) > xBound**2:
            integrand = 0.0
        else:
            integrand = \
                rhoPreFacs(x1, x1p, a, b, a4)**2 * \
                gFunc(x1, x1p, a, b)**2
        
        return integrand
    
    purityEst, purityErr = \
        scipy.integrate.dblquad(
        integrandFunc,
        -xBound, +xBound,
        -xBound, +xBound, 
        args=(a, b, a4))

    return purityEst, purityErr
################################################################################
