# This code is part of qcircha.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
We thanks D. Jaschke for providing this code

Variables:

mm : int, number of rows for mxn matrix
nn : int, number of columns for mxn matrix
sigma : float, standard deviation of normal distribution for entries of matrix
lam : float, ratio m / n

"""

import numpy as np
import numpy.linalg as nla
import matplotlib.pyplot as plt

__all__ = ['cumul_distr_datapoints', 'cumul_distr_svd_via_pdf', 'gen_mp']



import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity


def marchenko_pastur_pdf(var, lambda_ratio, number_datapoints):
    """
    The probability density function for the Marchenko-Pastur distribution
    is returned as grid for the eigenvalues and grid for the probabilities.

    **Arguments**

    var : float
        Variance of the underlying Gaussian distribution

    lambda_ratio : float
        Ratio of the number of rows over the number of columns in the
        random matrix.

    number_datapoints : int
        Gridpoints for eigenvalues.

    **Details**

    More information can be found on:

    * https://en.wikipedia.org/wiki/Marchenko-Pastur_distribution
    * https://medium.com/swlh/an-empirical-view-of-marchenko-pastur-theorem-1f564af5603d
    """
    lambda_minus = var * (1 - (1. / lambda_ratio)**0.5)**2
    lambda_plus = var * (1 + (1. / lambda_ratio)**0.5)**2

    ev_grid = np.linspace(lambda_minus, lambda_plus, number_datapoints)
    pdf = lambda_ratio / (2 * np.pi * var * ev_grid) * ((lambda_plus - ev_grid) * (ev_grid - lambda_minus))**0.5

    return ev_grid, pdf


def gen_mp(mm, nn, sigma, datapoints, samplepoints):
    """
    Returns singular values in descending order according to Marchenko Pastur
    distribution.
    """
    xgrid, ygrid = cumul_distr_svd_via_pdf(mm, nn, sigma, datapoints)

    vec = np.random.rand(samplepoints)
    vec = vec[np.argsort(vec)]

    idx = 0
    for ii in range(samplepoints):
        while(ygrid[idx + 1] < vec[ii]):
            idx += 1

        gr = (xgrid[idx + 1] - xgrid[idx]) / (ygrid[idx + 1] - ygrid[idx])
        vec[ii] = xgrid[idx] + gr * (vec[ii] - ygrid[idx])

    return vec[::-1]


def cumul_distr_svd_via_pdf(mm, nn, sigma, datapoints):
    xpdf, ypdf = marchenko_pastur_pdf(sigma, mm / nn, datapoints)

    xpdf = xpdf[1:]
    ypdf = ypdf[1:]

    cypdf = np.cumsum(ypdf)
    cypdf /= cypdf[-1]

    return np.sqrt(xpdf), cypdf


def cumul_distr_datapoints(mm, nn, sigma, datapoints=1000):
    lam, lam_plus, lam_minus = get_lambdas(mm, nn, sigma)
    xvec = np.linspace(0, lam_plus, datapoints)[1:-1]

    yvec = cumul_distr(xvec, mm, nn, sigma)

    return xvec, yvec


def cumul_distr(xvec, mm, nn, sigma):
    yvec = np.zeros(xvec.shape)

    lam, lam_plus, lam_minus = get_lambdas(mm, nn, sigma)

    if(lam > 1):
        mask_f1 = xvec < lam_minus
        mask_f2 = np.logical_and(xvec > lam_minus, xvec < lam_plus)
        mask_1 = xvec >= lam_plus

        yvec[mask_f1] = (lam - 1) / lam
        yvec[mask_f2] = (lam - 1) / (2 * lam) + ffunc(xvec[mask_f2], mm, nn, sigma)
        yvec[mask_1] = 1.0

    else:
        mask_f = np.logical_and(xvec > lam_minus, xvec < lam_plus)
        mask_1 = xvec >= lam_plus

        yvec[mask_f] = ffunc(xvec[mask_f], mm, nn, sigma)
        yvec[mask_1] = 1.0

    return yvec


def ffunc(xvec, mm, nn, sigma):
    # https://en.wikipedia.org/wiki/Marchenko%E2%80%93Pastur_distribution

    lam, lam_plus, lam_minus = get_lambdas(mm, nn, sigma)

    # prefactor
    c1 =  1. / (2 * np.pi * lam)

    # 1st term in sum
    s1 = np.pi * lam

    # 2nd term in sum
    s2 = np.sqrt((lam_plus - xvec) * (xvec - lam_minus)) / sigma**2

    # evaluate rfunc
    rvec = rfunc(xvec, mm, nn, sigma)

    # 3rd term
    s3 = - (1 + lam) * np.arctan((rvec**2 - 1) / 2 / rvec)

    # 4th term
    if(lam == 1):
        # lambda == 1 leads to a division by zero, which we have to handle
        # If abs(nominator) > 0, the division goes to infinity and the
        # value of the arctan for infinity is defined as +/- pi / 2
        s4 = np.zeros(rvec.shape)
        for ii, rii in enumerate(rvec):
            if(lam_minus * rii**2 - lam_plus == 0):
                raise Exception('Not handled')
            elif(lam_minus * rii**2 - lam_plus < 0):
                s4[ii] = - (1 - lam) * np.pi / 2
            else:
                s4[ii] = (1 - lam) * np.pi / 2
    else:
        tmp = (lam_minus * rvec**2 - lam_plus) / (2 * sigma**2 * (1 - lam) * rvec)
        s4 = (1 - lam) * np.arctan(tmp)

    return c1 * (s1 + s2 + s3 + s4)


def rfunc(xvec, mm, nn, sigma):
    lam, lam_plus, lam_minus = get_lambdas(mm, nn, sigma)

    return np.sqrt((lam_plus - xvec) / (xvec - lam_minus))


def get_lambdas(mm, nn, sigma):
    lam = mm / nn
    lam_plus = sigma**2 * (1 + np.sqrt(lam))**2
    lam_minus = sigma**2 * (1 - np.sqrt(lam))**2

    return lam, lam_plus, lam_minus


def main():
    l1 = 10
    l2 = 10
    sigma = np.sqrt(1.0 / 2**(l1 + l2))
    xvec, yvec = cumul_distr_datapoints(2**l1, 2**l2, sigma, datapoints=1000)
    #xvec = np.linspace(0, 1.1, 41)
    #yvec = yvec = cumul_distr(xvec, 2**8, 2**8, 5.0e-1)

    xpdf, ypdf = marchenko_pastur_pdf(sigma, 1.0, 201)
    print(xpdf)
    print(ypdf)
    xpdf = xpdf[1:]
    ypdf = ypdf[1:]
    ypdf /= np.sum(ypdf)
    cypdf = np.cumsum(ypdf)

    nn_iter = 20
    singvals = np.zeros((nn_iter, 2**min(l1, l2)))
    for jj in range(nn_iter):
        rand_mat = np.random.randn(2**l1 * 2**l2) * sigma
        norm = np.sum(rand_mat**2)
        print('Norm / dev', norm, abs(norm - 1))
        rand_mat /= np.sqrt(norm)
        rand_mat = np.reshape(rand_mat, [2**l1, 2**l2])

        singvals[jj, :] = nla.svd(rand_mat, compute_uv=False)

    singvals = singvals.flatten()
    singvals = singvals[np.argsort(singvals)]
    xgrid = np.cumsum(np.ones(singvals.shape)) / singvals.shape[0]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    #ax.plot(xvec, yvec, label='EV')
    #ax.plot(np.sqrt(xvec), yvec, label='SV')
    #ax.plot(singvals, xgrid, label='Rand')
    ax.plot(singvals**2, xgrid, label='Rand2')
    ax.plot(xpdf, cypdf, label='PDF')

    plt.legend(loc='lower left')
    plt.savefig('marchenko_pastur_cumulative_distribution.pdf')

    return

if __name__ == "__main__":
    main()