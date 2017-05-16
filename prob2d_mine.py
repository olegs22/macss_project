import numpy as np
import matplotlib.pyplot as plt
from hmf import load_hmf
from shmr import SHMR

def f_blue(Halo_mass, lg_mq, mu_c):
    #log10 halo_mass
    _mh = 10.0**(Halo_mass - lg_mq)
    return(np.exp(-_mh**mu_c))


def get_prob2d1(theta, use_red=True, sigma_lnMs=0.49730, hmfdata="../data/hmf.dat"):
    lg_mq, mu_c = theta
    """
    Derive p(lnMs, lnMh) by using the Bayes' Theorem:

        p(lnMs, lnMh) = p(lnMs | lnMh) * p(lnMh)

    """

    # the first step is to derive p(lnMh)
    # loading halo mass function (HMF)

    lnMh_arr, dndlnMh_arr = load_hmf(hmfdata)
    lg_mh1 = lnMh_arr / np.log(10.)

    blue_f = f_blue(lg_mh1, lg_mq, mu_c)
    red_f = 1. - blue_f
    if use_red:
        dndlnMh_arr = dndlnMh_arr * red_f
    else:
        dndlnMh_arr = dndlnMh_arr * blue_f


    # integrate over HMF to get the total number density of halos
    ntot = np.trapz(dndlnMh_arr, x=lnMh_arr)
    # normalize HMF to get p(lnMh)
    p_lnMh_arr = dndlnMh_arr / ntot
    #
    # the second step is to derive p(lnMs | lnMh)
    # getting mean log-stellar mass at fixed halo mass (assuming default parameters in the SHMR class)
    shmr_mean = SHMR()
    lnMs_mean_arr = shmr_mean.log_stellarmass_mean(lnMh_arr)
    #
    # make a 2d array for storing the p(lnMs | lnMh)
    Ms_arr = np.logspace(9, 13, 121)
    # Ms_arr = np.logspace(6, 13, 141)
    lnMs_arr = np.log(Ms_arr)
    nh = lnMh_arr.size
    ns = lnMs_arr.size
    # p(lnMs | lnMh)
    p_lnMs_at_lnMh_arr = np.zeros((ns, nh))
    #
    # calculate p(lnMs | lnMh) at fixed lnMh
    for i in xrange(nh):
        lnMs_mean = lnMs_mean_arr[i]
        p_lnMs_at_lnMh_arr[:, i] = gaussian(lnMs_arr, lnMs_mean, sigma_lnMs)
        _norm = np.trapz(p_lnMs_at_lnMh_arr[:, i], x=lnMs_arr)
        # plt.plot(lnMs_arr, p_lnMs_at_lnMh_arr[:, i], 'r-')
        # plt.show()
        if _norm == 0:
            p_lnMs_at_lnMh_arr[:, i] = 1e-300
        else:
            # the gaussian may be cuttoff at small Ms, make sure p_lnMs_at_lnMh_arr
            # is normalized to unity
            p_lnMs_at_lnMh_arr[:, i] /= _norm
    # the final step is to derive p(lnMs , lnMh)
    # make a 2d array for storing the 2d distribution of p(lnMs , lnMh)
    p2d_arr = np.zeros((ns, nh))
    for i in xrange(nh):
        p2d_arr[:, i] = p_lnMs_at_lnMh_arr[:, i] * p_lnMh_arr[i]
    return(p2d_arr, lnMs_arr, lnMh_arr)


def gaussian(x, mean, sigma):
    """Gaussian distribution with mean and sigma."""
    k = (x-mean)/sigma
    y = np.exp(-0.5*k**2)/sigma/np.sqrt(2.0*np.pi)
    return(y)


if __name__ == "__main__":
    p = [12.0, 0.3]
    for i in xrange(2):
        if i == 0:
            use_red = True
            ax = plt.subplot(211)
            xlabel = r"$\lg\;M_s^{\mathrm{red}}$"
        elif i == 1:
            use_red = False
            ax = plt.subplot(212)
            xlabel = r"$\lg\;M_s^{\mathrm{blue}}$"
        p2d_arr, lnMs_arr, lnMh_arr = get_prob2d_split(p, use_red=use_red)
        lgmhmin, lgmhmax = lnMh_arr.min()/np.log(10.), lnMh_arr.max()/np.log(10.)
        lgmsmin, lgmsmax = lnMs_arr.min()/np.log(10.), lnMs_arr.max()/np.log(10.)
        extent = [lgmhmin, lgmhmax, lgmsmin, lgmsmax]
        im = plt.imshow(p2d_arr, cmap='rainbow', origin='lower', extent=extent, vmin=0, vmax=0.01)
        plt.colorbar(im, orientation='vertical')
        plt.ylim(9.5, 12)
        plt.xlim(10., 15)
        plt.xlabel(r"$\lg\;M_h$", size=20)
        plt.ylabel(xlabel, size=20)
    plt.show()
