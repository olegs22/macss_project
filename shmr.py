# 10 May 2017 09:44:05

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline1d
import matplotlib.pyplot as plt


"""Stellar Mass to Halo Mass Relation (SHMR)"""

class SHMR(object):
    """ Stellar-Halo Mass Relationship.

    Parameters
    ----------

    Mh_1 : float
        Mh_1 is the characteristic halo mass.
    Ms_0 : float
        Ms_0 is characteristic stellar mass.
    beta : float
        beta is the low-mass power-law slope.
    delta : float
        delta is regulating how rapidly the relation climbs at high mass end.
    gamma : float
        gamma controls the transition from powerlaw to sub-exponential behavior.
    msmin : float, optional
        minimal stellar mass considered for the inverse function.
    msmax : float, optional
        maximal stellar mass considered for the inverse function.
    nmsbin : integer, optional
        number of bins in stellar mass considered for the inverse function.

    """
    def __init__(self, Mh_1=1.256e12, Ms_0=2.032e10, beta=0.33272, delta=0.440,
                 gamma=1.20579, msmin=5.e5, msmax=5.e15, nmsbin=100):
        """ spline requires a relatively large range of msmin+msmin.
        """
        # nodes for interpolation
        self.msarr = np.logspace(np.log10(msmin), np.log10(msmax), nmsbin)
        self.lgmsarr = np.log10(self.msarr)
        # parameters
        self.lgMh_1 = np.log10(Mh_1)
        self.lgMs_0 = np.log10(Ms_0)
        self.Ms_0 = Ms_0
        self.beta = beta
        self.delta = delta
        self.gamma = gamma
        # self.lg2ln = np.log(10.0)
        self.lg2ln = 2.3025850929940459
        # boundary of interpolation
        self.msmin = msmin
        self.msmax = msmax
        # to get log_stellarmass_mean
        self._inverse_lgMh()

    def _inverse_lgMh(self):
        """ I like natural log """
        self.lgmharr = self._lgMh(self.msarr, check_boundary=False)
        self.log_stellarmass_mean = spline1d(self.lgmharr*self.lg2ln, self.lgmsarr*self.lg2ln)

    def _lgMh(self, Ms, check_boundary=True):
        """ Eqn 2. in Leauthaud et al. 2011, mapping from
        Ms to Mh.
        """
        if check_boundary:
            # this works for both scalar and array
            if np.min(Ms) < self.msmin or np.max(Ms) > self.msmax:
                print self.msmin
                print self.msmax
                print np.min(Ms)
                print np.max(Ms)
                raise MsRangeError("input stellar mass range is unphysical for SMHR")
        _r = Ms/self.Ms_0
        # XXX the equation 19 in Zu and Mandelbaum 2015 has a typo, the exponential should be 10^
        _f = self.lgMh_1 + self.beta * (np.log10(Ms) - self.lgMs_0) + \
            _r**self.delta / (1.+_r**(-self.gamma)) - 0.5
        return(_f)

    def log_halomass_mean(self, lnMs):
        return(self._lgMh(np.exp(lnMs))*self.lg2ln)

if __name__ == "__main__":
    shmr = SHMR()
    Mh_arr = np.logspace(10, 16, 121)
    lnMh_arr = np.log(Mh_arr)
    # this is the median SHMR: lnMs = Func(lnMh)
    lnMs_arr = shmr.log_stellarmass_mean(lnMh_arr)
    # convert natural log to base-10 log
    lgMh_arr = lnMh_arr / np.log(10.0)
    lgMs_arr = lnMs_arr / np.log(10.0)
    plt.plot(lgMh_arr, lgMs_arr, 'r-')
    plt.xlabel(r"$M_h\;[M_{\odot}]$")
    plt.ylabel(r"$M_*\;[M_{\odot}/h^2]$")
    plt.show()
