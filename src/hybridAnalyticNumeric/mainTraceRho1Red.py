import numpy as np
import scipy
import scipy.special
import scipy.integrate
import matplotlib.pyplot as plt

import logging

from functions import \
        gFunc, fFunc, normbNeg00, rhoPreFacs, a4bNegFunc, \
        trPiece1, trPiece2, \
        purityPiece1, purityPiece2

logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s", level=logging.INFO)

################################################################################
if __name__ == '__main__':
    a2idpiov4 = 1

    bList = []
    normList = []
    a4List = []

    trRhoRedPiece1EstList = []
    trRhoRedPiece2EstList = []

    trRhoRedPiece1ErrList = []
    trRhoRedPiece2ErrList = []

    trRhoRedEstList = []
    trRhoRedErrList = []

    purityPiece1EstList = []
    purityPiece2EstList = []

    purityPiece1ErrList = []
    purityPiece2ErrList = []

    purityEstList = []
    purityErrList = []

    for i in range(0, 141):

        b2idpiov4 = -(i + 1) * 0.25
        logging.info("Begin analysis for b = {0}".format(b2idpiov4))

        if abs(b2idpiov4/a2idpiov4) < 1:
            INF_APPROX = \
                 4*np.sqrt(3/2*abs(a2idpiov4/b2idpiov4))

        else:
            INF_APPROX = \
                 4*np.sqrt(2/3*abs(b2idpiov4/a2idpiov4))

        if b2idpiov4 < 0:
            norm = normbNeg00(a2idpiov4, b2idpiov4)
            a4 = a4bNegFunc(a2idpiov4, b2idpiov4)
            normList.append(norm)
            a4List.append(a4)
        elif b2idpiov4 > 0:
            pass
        elif b2idpiov4 == 0:
            pass

        trRhoRedPiece1Est, trRhoRedPiece1Err = \
            trPiece1(a2idpiov4, b2idpiov4, a4, INF_APPROX)

        trRhoRedPiece2Est, trRhoRedPiece2Err = \
            trPiece2(a2idpiov4, b2idpiov4, a4)

        trRhoRedEst = trRhoRedPiece1Est + trRhoRedPiece2Est

        trRhoRedErr = np.sqrt(trRhoRedPiece1Err**2 + trRhoRedPiece2Err**2)

        logging.info("Probability double-check for b = {0}".format(b2idpiov4))
        logging.info("that trace rho reduced == 1")
        logging.info("trRhoRedPiece1 = %g +- %g" % (trRhoRedPiece1Est, trRhoRedPiece1Err))
        logging.info("trRhoRedPiece2 = %g +- %g" % (trRhoRedPiece2Est, trRhoRedPiece2Err))
        logging.info("trRhoRed = %g +- %g" % (trRhoRedEst, trRhoRedErr))

        purityPiece1Est, purityPiece1Err = \
            purityPiece1(a2idpiov4, b2idpiov4, a4, INF_APPROX)

        purityPiece2Est, purityPiece2Err = \
            purityPiece2(a2idpiov4, b2idpiov4, a4)

        purityEst = purityPiece1Est + purityPiece2Est
        purityErr = np.sqrt(purityPiece1Err**2 + purityPiece2Err**2)

        logging.info("Purity for b = %g" % (b2idpiov4))
        logging.info("purityPiece1 = %g +- %g" % (purityPiece1Est, purityPiece1Err))
        logging.info("purityPiece2 = %g +- %g" % (purityPiece2Est, purityPiece2Err))
        logging.info("purity = %g +- %g" % (purityEst, purityErr))
        
        bList.append(b2idpiov4)

        trRhoRedPiece1EstList.append(trRhoRedPiece1Est)
        trRhoRedPiece2EstList.append(trRhoRedPiece2Est)

        trRhoRedPiece1ErrList.append(trRhoRedPiece1Err)
        trRhoRedPiece2ErrList.append(trRhoRedPiece2Err)

        trRhoRedEstList.append(trRhoRedEst)
        trRhoRedErrList.append(trRhoRedErr)

        purityPiece1EstList.append(purityPiece1Est)
        purityPiece2EstList.append(purityPiece2Est)

        purityPiece1ErrList.append(purityPiece1Err)
        purityPiece2ErrList.append(purityPiece2Err)

        purityEstList.append(purityEst)
        purityErrList.append(purityErr)

    fig, axes = plt.subplots(1, 2, figsize=(34.58/2.54, 13.8/2.54))

    axNorm = axes[0]
    axA4 = axes[1]

    axNormInset = axNorm.inset_axes([0.2, 0.2, 0.6, 0.6])
    axA4Inset = axA4.inset_axes([0.2, 0.2, 0.6, 0.6])

    axNorm.plot(bList, normList)
    axA4.plot(bList, a4List)

    axNormInset.plot(bList, normList)
    axA4Inset.plot(bList, a4List)

    for ax in axes:
        ax.set_xlabel(r'$b$')

    axNormInset.set_yscale('log')
    axA4Inset.set_yscale('log')

    axNorm.set_ylabel(r'normbneg00')
    axA4.set_ylabel(r'$A^4$')
    plt.savefig("normPlots.pdf")

    plt.close()
        
    fig, axes = plt.subplots(
        2, 3, figsize=(34.58/2.5, 13.8/2.54),
        constrained_layout=True)

    axes[0, 0].errorbar(bList, trRhoRedPiece1EstList, trRhoRedPiece1ErrList,
                        marker='s', capsize=4,fillstyle='none')
    axes[0, 1].errorbar(bList, trRhoRedPiece2EstList, trRhoRedPiece2ErrList,
                        marker='s', capsize=4, fillstyle='none')
    axes[0, 2].errorbar(bList, trRhoRedEstList, trRhoRedErrList,
                        marker='s', capsize=4, fillstyle='none')

    axes[0, 0].set_title('trPiece1')
    axes[0, 1].set_title('trPiece2')
    axes[0, 2].set_title('tr')

    axes[1, 0].errorbar(bList, purityPiece1EstList, purityPiece1ErrList,
                        marker='o', capsize=4, fillstyle='none')
    axes[1, 1].errorbar(bList, purityPiece2EstList, purityPiece2ErrList,
                        marker='o', capsize=4, fillstyle='none')
    axes[1, 2].errorbar(bList, purityEstList, purityErrList,
                        marker='o', capsize=4, fillstyle='none')

    axes[1, 0].set_title('purityPiece1')
    axes[1, 1].set_title('purityPiece2')
    axes[1, 2].set_title('purity')

    
    for ax in axes[0, :]:
        ax.set_xlabel(r'$b$')
        ax.set_ylabel(r'$\mathrm{tr}\left(\rho_{\mathrm{reduced}}\right)$')
        ax.axhline(0.0, color='k', alpha=0.5, ls='dashed')
        ax.axhline(0.5, color='k', alpha=0.5, ls='dashed')
        ax.axhline(1.0, color='k', alpha=0.5, ls='dashed')


    for ax in axes[1, :]:
        ax.set_xlabel(r'$b$')
        ax.set_ylabel(r'$\mathrm{tr}\left(\rho_{\mathrm{reduced}}^2\right)$')
        ax.axhline(0.0, color='k', alpha=0.5, ls='dashed')
        ax.axhline(0.5, color='k', alpha=0.5, ls='dashed')
        ax.axhline(1.0, color='k', alpha=0.5, ls='dashed')

    plt.savefig('tracePlots.pdf')
    plt.close()
